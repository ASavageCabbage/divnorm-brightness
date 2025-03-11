import os

import cv2
import numpy as np


# Needed to use OpenEXR with OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Convert sRGB image to linear luminance.

    Requires input to be in the range [0, 1].
    """
    assert np.max(image) <= 1
    result = ((image + 0.055) / 1.055) ** 2.4
    linear_mask = image <= 0.04045
    result[linear_mask] = image[linear_mask] / 12.92
    return result


def _linear_to_srgb(image: np.ndarray):
    """Convert linear RGB image to sRGB.

    Requires input to be in the range [0, 1].
    """
    assert np.max(image) <= 1
    result = 1.055 * (image ** (1 / 2.4)) - 0.055
    linear_mask = image <= 0.0031308
    result[linear_mask] = image[linear_mask] * 12.92
    return result


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
        image = _srgb_to_linear(image / 255)

    return image


def downsample_image(
    image: np.ndarray, resize_width: int = None, resize_height: int = None
) -> np.ndarray:
    """Downsample image if requested resize width or height is less than original."""
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
        if height > resize_height or width > resize_width:
            image = cv2.resize(
                image, (resize_width, resize_height), interpolation=cv2.INTER_AREA
            )
    return image


def write_image(file_path: str, image: np.ndarray):
    """Write linear RGB image array to file_path as an SDR image.

    image must have dimensions (w, h, 3).
    Output image will be clipped to (0, 1) and coverted to sRGB.
    """
    image = _linear_to_srgb(np.clip(image, 0, 1))
    image = (image * 255).astype(np.int32)
    # RGB to BGR
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    assert cv2.imwrite(file_path, np.stack([b, g, r], axis=-1))


def generate_scales(width: int, cs_ratio: float, num_scales: int) -> list[float]:
    """Generate an exponential list of scales up to width * cs_ratio."""
    scale = width * cs_ratio
    scales = []
    for _ in range(num_scales):
        scales.insert(0, scale)
        scale /= cs_ratio
    return scales
