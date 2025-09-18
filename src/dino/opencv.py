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
        borderType=cv2.BORDER_REFLECT_101,
    )
    return np.squeeze(blurred)


def generate_mipmap(
    image: np.ndarray, downscale_ratio: float, levels: int
) -> list[np.ndarray]:
    images = [image]
    for i in range(levels):
        prev_image = images[-1]
        height, width = prev_image.shape[:2]
        new_w = max(1, round(width / downscale_ratio))
        new_h = max(1, round(height / downscale_ratio))
        images.append(
            cv2.resize(prev_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        )
    return images


def upscale_from_mipmap(
    mipmap: list[np.ndarray], level: int, interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    height, width = mipmap[0].shape[:2]
    return (
        mipmap[level]
        if level == 0
        else cv2.resize(mipmap[level], (width, height), interpolation=interpolation)
    )


def efficient_blur_from_mipmap(
    mipmap: list[np.ndarray], level: int, sigma: int
) -> np.ndarray:
    height, width = mipmap[0].shape[:2]
    blurred = gaussian_blur(mipmap[level], sigma)
    return cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)


def gaussian_blur_all_scales(
    image: np.ndarray, cs_ratio: float, num_scales: int
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Compute and return a Gaussian-blurred image for all envelope scales.

    Returns a generator of (<sigma>, <Gaussian-blurred image>) tuples for all scales.
    """
    mipmap = generate_mipmap(image, cs_ratio, num_scales)
    for i, scale in enumerate(
        generate_scales(np.max(image.shape), cs_ratio, num_scales)
    ):
        yield scale, efficient_blur_from_mipmap(mipmap, i, cs_ratio)


def arcmin2_to_pixel2(value: float, width_pixels: int, fov_degrees: float) -> float:
    """Convert a value in arcmin-squared to pixels-squared using small-angle approximation."""
    return value * (width_pixels / (60 * fov_degrees)) ** 2


def dinos_upscale_only(
    L: np.ndarray,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
    w: float = 0.9,
    d_nit_arcmin2: float = 100,
    image_fov_degrees: float = 72,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    d = arcmin2_to_pixel2(d_nit_arcmin2, L.shape[1], image_fov_degrees)
    weights = [w**i for i in range(num_scales)]
    weighted_sum = np.zeros(L.shape[:2])
    scales = generate_scales(L.shape[1], cs_ratio, num_scales)
    mipmap = generate_mipmap(L, cs_ratio, num_scales)
    center_response = upscale_from_mipmap(mipmap, 0, interpolation=interpolation)

    for i in range(1, num_scales):
        surround_response = upscale_from_mipmap(mipmap, i, interpolation=interpolation)
        _d = d / scales[i] ** 2
        weighted_sum += weights[i] * (
            (center_response + _d) / (surround_response + _d) - 1
        )
        i += 1
        center_response = surround_response

    return weighted_sum


def dinos_efficient(
    L: np.ndarray,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
    w: float = 0.9,
    d_nit_arcmin2: float = 100,
    image_fov_degrees: float = 72,
) -> np.ndarray:
    d = arcmin2_to_pixel2(d_nit_arcmin2, L.shape[1], image_fov_degrees)
    weights = [w**i for i in range(num_scales)]
    # Initialize weighted_sum as a 2D array
    weighted_sum = np.zeros(L.shape[:2])
    # Compute ratios and weighted sum using only two blurred images at a time
    center_response = None

    i = 0
    for scale, surround_response in gaussian_blur_all_scales(L, cs_ratio, num_scales):
        if center_response is not None:
            _d = d / scale**2
            weighted_sum += weights[i] * (
                (center_response + _d) / (surround_response + _d) - 1
            )
            i += 1
        center_response = surround_response

    return weighted_sum
