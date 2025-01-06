import os
from typing import Generator

import cv2
import numpy as np

from src.dino.util import *


# Needed to use OpenEXR with OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def resize_image(
    image: np.ndarray, resize_width: int = None, resize_height: int = None
) -> np.ndarray:
    (
        height,
        width,
    ) = image.shape[:2]
    if not (resize_width is None and resize_height is None):
        if resize_width is None:
            resize_width = resize_height / height * width
        if resize_height is None:
            resize_height = resize_width / width * height
        resize_width = round(resize_width)
        resize_height = round(resize_height)
        # Only resize down
        if height > resize_height or width > resize_width:
            image = cv2.resize(
                image, (resize_width, resize_height), interpolation=cv2.INTER_AREA
            )
    return image


def read_image(file_path: str, min_value: float = 0) -> np.ndarray:
    """Load HDR or SDR image from disk as RGB numpy array.

    Returns image with linear sRGB channels (or unchanged in the case of HDR images).
    """
    bgr_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # RGB or RGBA only
    assert len(bgr_image.shape) == 3
    assert bgr_image.shape[2] >= 3
    # Negative values shouldn't exist
    bgr_image[bgr_image < min_value] = min_value
    # OpenCV uses BGR
    b = bgr_image[:, :, 0]
    g = bgr_image[:, :, 1]
    r = bgr_image[:, :, 2]
    image = np.stack([r, g, b], axis=-1)

    if not any(file_path.endswith(ext) for ext in [".hdr", ".exr"]):
        # uint8 for SDR images (0, 255)
        image = srgb_to_linear(image / 255)

    return image


def write_image(file_path: str, image: np.ndarray):
    """Write linear RGB image array to file_path as an SDR image.

    image must have dimensions (w, h, 3).
    Output image will be clipped to (0, 1) and coverted to sRGB.
    """
    image = linear_to_srgb(np.clip(image, 0, 1))
    image = (image * 255).astype(np.int32)
    # RGB to BGR
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    assert cv2.imwrite(file_path, np.stack([b, g, r], axis=-1))


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Convert sRGB image to linear luminance.

    Requires input to be in the range [0, 1].
    """
    assert np.max(image) <= 1
    result = ((image + 0.055) / 1.055) ** 2.4
    linear_mask = image <= 0.04045
    result[linear_mask] = image[linear_mask] / 12.92
    return result


def linear_to_srgb(image: np.ndarray):
    """Convert linear RGB image to sRGB.

    Requires input to be in the range [0, 1].
    """
    assert np.max(image) <= 1
    result = 1.055 * (image ** (1 / 2.4)) - 0.055
    linear_mask = image <= 0.0031308
    result[linear_mask] = image[linear_mask] * 12.92
    return result


def rgb_to_relative_luminance(image: np.ndarray) -> np.ndarray:
    """Convert linear sRGB to relative luminance.

    NOTE: Use srgb_to_linear to convert non-linear sRGB images to linear.
    """
    return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]


def fairchild_to_relative_luminance(image: np.ndarray) -> np.ndarray:
    """Convert Fairchild RGB to relative luminance."""
    return 0.1904 * image[:, :, 0] + 0.7646 * image[:, :, 1] + 0.0450 * image[:, :, 2]


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


def dn_brightness_model(
    L: np.ndarray,
    gamma: float = 2.2,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
    w: float = 0.9,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 1.0,
    scale_normalized_constants: bool = True,
) -> np.ndarray:
    """Apply divisive normalization brightness model to array of linear luminances.

    B(x,y) = sum(
        w**i * (
            (a*center + b / scales[i]**2) / (c*surround + d / scales[i]**2)
            - (b / scales[i]**2) / (d / scales[i]**2)
        )
    )

    NOTE: The current scale is the center stdev at the current scale,
          the next scale up is the surround stdev at the current scale.
    """
    L = scale_gamma(L, gamma=gamma)

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
            (a * center_response + _b) / (c * surround_response + _d) - _b / _d
        )
        center_response = surround_response

    return weighted_sum


def gaussian_blur_all_scales(
    L: np.ndarray,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Compute and return a Gaussian-blurred image for all envelope scales.

    Returns a generator of (<stdev>, <Gaussian-blurred image>) tuples for all scales.
    """
    for scale in generate_scales(L.shape[0], cs_ratio, num_scales):
        yield scale, gaussian_blur(L, scale)
