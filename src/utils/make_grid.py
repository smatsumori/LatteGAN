import numpy as np

import torch
from torchvision.utils import make_grid


def make_grid_from_numpy(images, nrow=5, normalize=lambda x: x):
    """Tile batch images of numpy as one grid image.

    Args:
        images (np.ndarray): shape (B, C, H, W).
        nrow (int, optional): number of images per a row. Defaults to 5.
        normalize (Callable, optional): Normalize image function, value range (0, 1). Defaults to lambdax:x.

    Returns:
        np.ndarray: tiled image shape (H', W', C).
    """
    images = normalize(images)
    images = np.clip(images, 0., 1.)

    images = torch.from_numpy(images)
    grid = make_grid(images, nrow=nrow)
    grid = grid.numpy()

    grid = np.transpose(grid, (1, 2, 0))
    return grid
