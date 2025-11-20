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
    """A small training loop used by the notebook examples.

    This function is intentionally minimal; it assumes `x_init` contains
    the dataset (first dimension = data_size) and that `model` accepts
    `(x_noisy, t)` returning predicted noise.
    """
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(data_size)
        epoch_loss = 0.0

        for i in range(0, data_size, batch_size):
            indices = perm[i : i + batch_size]
            x_batch = x_init[indices]

            # For each item in the batch, sample a random t and apply noising
            t = torch.randint(1, T + 1, (len(x_batch),)).to(device)
            x_noisy = torch.stack([
                noise_x_t(x.unsqueeze(0), t_i.item(), sigma_list).squeeze(0)
                for x, t_i in zip(x_batch, t)
            ])

            optimizer.zero_grad()
            predicted_noise = model(x_noisy, t)
            loss = loss_fct(predicted_noise, x_batch - x_noisy)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / (data_size / batch_size)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
