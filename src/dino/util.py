"""Common functions that work in both OpenCV and PyTorch implementations."""

import numpy as np
import torch


def xyz_to_lxy(
    X: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
    Z: np.ndarray | torch.Tensor,
) -> tuple[
    np.ndarray | torch.Tensor,
    np.ndarray | torch.Tensor,
    np.ndarray | torch.Tensor,
]:
    """Convert from XYZ to LXY colour space."""
    denom = X + Y + Z + 1e-8  # Prevent division by zero
    x_chroma = X / denom
    y_chroma = Y / denom
    return Y, x_chroma, y_chroma  # Use Y as luminance


def scale_gamma(
    image: np.ndarray | torch.Tensor,
    gamma: float = 2.2,
) -> np.ndarray | torch.Tensor:
    """Convert linear RGB to nonlinear RGB space.

    Expects input channels to be in the range [0, 1].
    """
    # TODO: replace with sRGB transfer function?
    return image ** (1.0 / gamma)


def generate_scales(width: int, cs_ratio: float, num_scales: int) -> list[float]:
    """Generate an exponential list of scales up to width * cs_ratio."""
    scale = width * cs_ratio
    scales = []
    for _ in range(num_scales):
        scales.insert(0, scale)
        scale /= cs_ratio
    return scales
