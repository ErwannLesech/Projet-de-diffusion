"""Utility helpers used by the notebooks and demos.

Small, focused utilities for image-to-point conversion and simple
packing/unpacking of 2D coordinate lists.
"""

from typing import List, Tuple
from PIL import Image

IMG_SIZE = 150


def scatter_pixels(img_file: str) -> Tuple[List[int], List[int]]:
    """Return two lists (x_coords, y_coords) of black pixels from image.

    The image is converted to binary (black/white) and resized to
    `IMG_SIZE` x `IMG_SIZE` before extracting coordinates.
    """
    w = IMG_SIZE
    img = Image.open(img_file).resize((w, w)).convert("1")
    pels = img.load()
    black_pels = [(x, y) for x in range(w) for y in range(w) if pels[x, y] == 0]
    # invert Y coordinate so images map to Cartesian-like coordinates
    return [t[0] for t in black_pels], [w - t[1] for t in black_pels]


def pack_data(x: List[float], y: List[float]) -> List[float]:
    """Interleave two lists `x` and `y` into a single 1D list.

    Result format: [x0, y0, x1, y1, ...].
    """
    return [val for pair in zip(x, y) for val in pair]


def unpack_1d_data(one_d_data: List[float]) -> Tuple[List[float], List[float]]:
    """Split an interleaved 1D list into two lists (x, y).

    Expects input like [x0, y0, x1, y1, ...].
    """
    x = one_d_data[0::2]
    y = one_d_data[1::2]
    return x, y


