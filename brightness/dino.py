import os
from typing import Generator

import cv2
import numpy as np


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
                image, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC
            )
    return image


def read_image(
    file_path: str, white_nits: float = 400, gamma: float = 2.2
) -> np.ndarray:
    """Load HDR or SDR image from disk as RGB numpy array.

    Assumes EXR images are in units of nits (candela/m^2).
    Assumes SDR images are displayed on a monitor at the specified luminance (nits) for pure white.

    Returns image with RGB channels in nits (linear luminances).
    """
    bgr_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # RGB or RGBA only
    assert len(bgr_image.shape) == 3
    assert bgr_image.shape[2] >= 3
    # Negative values shouldn't exist
    bgr_image[bgr_image < 0] = 0
    # OpenCV uses BGR
    b = bgr_image[:, :, 0]
    g = bgr_image[:, :, 1]
    r = bgr_image[:, :, 2]
    image = np.stack([r, g, b], axis=-1)

    if not any(file_path.endswith(ext) for ext in [".hdr", ".exr"]):
        # uint8 for SDR images (0, 255)
        image = (image / 255) ** gamma
        image *= white_nits

    return image


def write_image(file_path: str, image: np.ndarray, gamma: float = 2.2):
    """Write an SDR RGB image array to file_path.

    image must have dimensions (w, h, 3) and take values from (0, 1).
    Values less than 0 will be clipped to 0.
    Values greater than 1 will be clipped to 1.
    """
    image[image < 0] = 0
    image[image > 1] = 1
    # Use nonlinear luminance for SDR
    image = image ** (1.0 / gamma)
    image = (image * 255).astype(np.int32)
    # RGB to BGR
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    cv2.imwrite(file_path, np.stack([b, g, r], axis=-1))


def rgb_to_xyz(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Conversion matrix from sRGB to XYZ
    rgb_to_xyz_matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = image @ rgb_to_xyz_matrix.T
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]  # X, Y, Z


def xyz_to_lxy(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denom = X + Y + Z + 1e-8  # Prevent division by zero
    x_chroma = X / denom
    y_chroma = Y / denom
    return Y, x_chroma, y_chroma  # Use Y as luminance


def lxy_to_rgb(
    luminance: np.ndarray, x_chroma: np.ndarray, y_chroma: np.ndarray
) -> np.ndarray:
    # Ensure all inputs are 2D arrays
    luminance = np.squeeze(luminance)
    x_chroma = np.squeeze(x_chroma)
    y_chroma = np.squeeze(y_chroma)

    # Validate shapes
    assert luminance.shape == x_chroma.shape
    assert luminance.shape == y_chroma.shape

    z_chroma = 1 - x_chroma - y_chroma
    X = luminance * x_chroma / (y_chroma + 1e-8)
    Z = luminance * z_chroma / (y_chroma + 1e-8)
    Y = luminance
    xyz_to_rgb_matrix = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    rgb = np.dot(np.stack([X, Y, Z], axis=-1), xyz_to_rgb_matrix.T)
    rgb = np.clip(rgb, 0, 1)  # Clip to valid range
    return rgb


def gaussian_blur_cv(
    image: np.ndarray, sigma: float, kernel_sigmas: int = 1
) -> np.ndarray:
    # Ensure the kernel size is odd
    kernel_size = int(2 * kernel_sigmas * sigma + 1) | 1
    blurred = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    return np.squeeze(blurred)


def generate_scales(
    shape: tuple[int, int], cs_ratio: float, min_scale: float
) -> list[float]:
    """Generate an exponential list of scales from max(shape) / cs_ratio to min_scale."""
    max_scale = np.min(shape)
    scale = max_scale / cs_ratio
    scales = []
    while scale >= min_scale:
        scales.insert(0, scale)
        scale /= cs_ratio
    return scales


def dn_brightness_model(
    L: np.ndarray,
    gamma: float = 2.2,
    cs_ratio: float = 2.0,
    min_scale: float = 1.0,
    w: float = 0.85,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 1.0,
    scale_normalized_constants: bool = False,
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
    L = L * (1.0 / gamma)

    scales = generate_scales(L.shape, cs_ratio, min_scale)
    weights = [w**i for i in range(len(scales))]

    # Initialize weighted_sum as a 2D array
    weighted_sum = np.zeros(L.shape[:2])

    # Compute ratios and weighted sum using only two blurred images at a time
    center_response = gaussian_blur_cv(L, scales[0])

    for i in range(1, len(scales)):
        surround_response = gaussian_blur_cv(L, scales[i])

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
    min_scale: float = 1.0,
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Compute and return a Gaussian-blurred image for all envelope scales.

    Returns a generator of (<stdev>, <Gaussian-blurred image>) tuples
    for all scales down to min_scale.
    """
    for scale in generate_scales(L.shape, cs_ratio, min_scale):
        yield scale, gaussian_blur_cv(L, scale)
