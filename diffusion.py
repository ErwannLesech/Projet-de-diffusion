"""Utilities for applying a simple diffusion (noising) process.

This module provides helper functions used in the project's notebook
and experiments. Changes in this commit are small documentation and
clarity improvements to make the functions easier to reuse.
"""

import math
from typing import Sequence

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def noise_x_t_with_t_minus_1(x_t_minus_1: torch.Tensor, t: int, sigma_list: Sequence[float]) -> torch.Tensor:
    """Return x_t given x_{t-1} using provided sigma list.

    Parameters
    - x_t_minus_1: tensor at time t-1
    - t: target time step (int)
    - sigma_list: sequence/array of sigma values (length at least t)

    If t == 0 the input is returned unchanged.
    """
    if t == 0:
        return x_t_minus_1
    sigma = float(sigma_list[t - 1])
    noise = torch.randn_like(x_t_minus_1).to(device)
    x_t = math.sqrt(sigma) * x_t_minus_1 + math.sqrt(1.0 - sigma) * noise
    return x_t


def noise_x_t(x_0: torch.Tensor, t: int, sigma_list: Sequence[float]) -> torch.Tensor:
    """Return x_t obtained from x_0 after t steps of the noising schedule.

    Uses the cumulative product of the `sigma_list` to compute the effective
    scale at time `t`.
    """
    alpha_prod = float(np.cumprod(sigma_list, axis=0)[t])
    noise = torch.randn_like(x_0).to(device)
    x_t = math.sqrt(alpha_prod) * x_0 + math.sqrt(1.0 - alpha_prod) * noise
    return x_t


def train(epochs: int, batch_size: int, model, optimizer, loss_fct, data_size: int, x_init: torch.Tensor, T: int, sigma_list: Sequence[float]):
    """Training loop following Algorithm 1 from the project specification.
    
    Implements the exact algorithm described in the LaTeX document:
    1. Select x_0 ~ p(x_0)
    2. Randomly draw t ~ U({1, ..., T})
    3. Randomly draw ξ_0 ~ N(0, I)
    4. Calculate x_t = sqrt(α̅_t)x_0 + sqrt(1-α̅_t)ξ_0
    5. Gradient descent on ||ξ_0 - ξ_θ(x_t, t)||_2^2
    
    Args:
        epochs: Number of training epochs
        batch_size: Size of mini-batches
        model: Neural network that predicts noise ξ_θ(x_t, t)
        optimizer: Optimizer for the model parameters
        loss_fct: Loss function (should be MSE)
        data_size: Size of the dataset
        x_init: Dataset tensor (first dimension = data_size)
        T: Number of diffusion timesteps
        sigma_list: Sequence of sigma values for the diffusion schedule
    """
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(data_size)
        epoch_loss = 0.0

        for i in range(0, data_size, batch_size):
            indices = perm[i : i + batch_size]
            x_batch = x_init[indices]  # Step 1: Select x_0 ~ p(x_0)

            # Step 2: Randomly draw t ~ U({1, ..., T})
            t = torch.randint(1, T + 1, (len(x_batch),)).to(device)
            
            # Step 3: Randomly draw ξ_0 ~ N(0, I) - IMPORTANT: we need to store this noise!
            noise_batch = torch.randn_like(x_batch).to(device)
            
            # Step 4: Calculate x_t = sqrt(α̅_t)x_0 + sqrt(1-α̅_t)ξ_0
            x_noisy_batch = []
            for j, (x_0, t_i, noise) in enumerate(zip(x_batch, t, noise_batch)):
                # Use cumulative product α̅_t = ∏_{i=0}^{t-1} σ_i
                t_idx = int(t_i.item()) - 1  # Convert to int for indexing
                alpha_bar_t = float(np.cumprod(sigma_list, axis=0)[t_idx])
                x_t = math.sqrt(alpha_bar_t) * x_0 + math.sqrt(1.0 - alpha_bar_t) * noise
                x_noisy_batch.append(x_t)
            
            x_noisy_batch = torch.stack(x_noisy_batch)

            optimizer.zero_grad()
            
            # Model predicts the noise ξ_θ(x_t, t)
            predicted_noise = model(x_noisy_batch, t)
            
            # Step 5: Gradient descent on ||ξ_0 - ξ_θ(x_t, t)||_2^2
            loss = loss_fct(predicted_noise, noise_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (data_size // batch_size)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")