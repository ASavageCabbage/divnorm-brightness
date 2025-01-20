import cv2
import numpy as np


def gaussian_blur(
    image: np.ndarray, sigma: float, kernel_sigmas: int = 2
) -> np.ndarray:
    # Ensure the kernel size is odd
    kernel_size = int(2 * kernel_sigmas * sigma + 1) | 1
    blurred = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_DEFAULT,
    )
    return np.squeeze(blurred)


def generate_scales(width: int, cs_ratio: float, num_scales: int) -> list[float]:
    """Generate an exponential list of scales up to width * cs_ratio."""
    scale = width * cs_ratio
    scales = []
    for _ in range(num_scales):
        scales.insert(0, scale)
        scale /= cs_ratio
    return scales


def dinos(
    L: np.ndarray,
    gamma: float = 2.2,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
    w: float = 0.9,
    b: float = 1.0,
    d: float = 1.0,
    scale_normalized_constants: bool = True,
) -> np.ndarray:
    """Apply divisive normalization brightness model to array of linear luminances.

    B(x,y) = sum(
        w**i * (
            (center + b / scales[i]**2) / (surround + d / scales[i]**2)
            - (b / scales[i]**2) / (d / scales[i]**2)
        )
    )

    NOTE: The current scale is the center stdev at the current scale,
          the next scale up is the surround stdev at the current scale.
    """
    L = L ** (1.0 / gamma)

    scales = generate_scales(L.shape[1], cs_ratio, num_scales)
    weights = [w**i for i in range(len(scales))]

    # Initialize weighted_sum as a 2D array
    weighted_sum = np.zeros(L.shape[:2])

    # Compute ratios and weighted sum using only two blurred images at a time
    center_response = gaussian_blur(L, scales[0])

    for i in range(1, len(scales)):
        surround_response = gaussian_blur(L, scales[i])

        assert center_response.ndim == 2
        assert surround_response.ndim == 2

        _b = b
        _d = d
        if scale_normalized_constants:
            _b /= scales[i] ** 2
            _d /= scales[i] ** 2

        weighted_sum += weights[i - 1] * (
            (center_response + _b) / (surround_response + _d) - _b / _d
        )
        center_response = surround_response

    return weighted_sum
