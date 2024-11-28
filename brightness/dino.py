import sys

import cv2
import numpy as np
import pyexr
import matplotlib.pyplot as plt
from PIL import Image


def read_image(file_path: str) -> np.ndarray:
    """Load an EXR or PNG image from disk as RGB numpy array."""
    if file_path.endswith(".exr"):
        gamma = 2.2
        exr_image = pyexr.open(file_path)
        # Check for individual "R", "G", "B" channels
        if (
            "R" in exr_image.channel_map
            and "G" in exr_image.channel_map
            and "B" in exr_image.channel_map
        ):
            r = exr_image.get("R")
            g = exr_image.get("G")
            b = exr_image.get("B")
            height, width, channels = r.shape
            if width > 4000:
                scale_factor = 0.5
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                r = cv2.resize(
                    r, (new_width, new_height), interpolation=cv2.INTER_CUBIC
                )
                g = cv2.resize(
                    g, (new_width, new_height), interpolation=cv2.INTER_CUBIC
                )
                b = cv2.resize(
                    b, (new_width, new_height), interpolation=cv2.INTER_CUBIC
                )
            image = np.stack([r, g, b], axis=-1)
            r = np.absolute(r)
            g = np.absolute(g)
            b = np.absolute(b)
            r = np.power(r, 1.0 / gamma)
            g = np.power(g, 1.0 / gamma)
            b = np.power(b, 1.0 / gamma)
        else:
            raise ValueError(
                f"EXR file does not contain R, G, B channels. Found: {exr_image.channel_map.keys()}"
            )
        return image.astype(np.float32)
    elif file_path.endswith(".png"):
        image = (
            np.asarray(Image.open(file_path).convert("RGB"), dtype=np.float32) / 255.0
        )
        return image
    else:
        raise ValueError(
            "Unsupported file format. Only EXR and PNG files are supported."
        )


def rgb_to_xyz(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Conversion matrix from sRGB to XYZ
    rgb_to_xyz_matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = np.dot(image, rgb_to_xyz_matrix.T)
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
    if luminance.shape != x_chroma.shape or luminance.shape != y_chroma.shape:
        raise ValueError(
            f"Shape mismatch: luminance {luminance.shape}, x_chroma {x_chroma.shape}, y_chroma {y_chroma.shape}"
        )

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


def gaussian_blur_cv(image: np.ndarray, sigma: float) -> np.ndarray:
    # Use OpenCV's GaussianBlur
    kernel_size = int(6 * sigma + 1) | 1  # Ensure the kernel size is odd
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )


def dn_brightness_model(
    L: np.ndarray,
    scales: list[float],
    weight_decay: float = 0.85,
    base_activation: float = 1e-8,
) -> np.ndarray:
    """Apply divisive normalization brightness model to array of luminances (i.e. greyscale input image).

    B(x,y) = sum(weight_decay**(i) * center / (surround + base_activation/scales[i]**2))

    NOTE: The current scale is the center stdev at the current scale,
          the next scale up is the surround stdev at the current scale.
    """
    weights = [weight_decay**i for i in range(len(scales))]

    # Initialize weighted_sum as a 2D array
    weighted_sum = np.zeros(L.shape[:2], dtype=L.dtype)

    # Compute ratios and weighted sum using only two blurred images at a time
    prev_blurred = gaussian_blur_cv(L, scales[0])
    prev_blurred = np.squeeze(prev_blurred)  # Ensure 2D

    for i in range(1, len(scales)):
        curr_blurred = gaussian_blur_cv(L, scales[i])
        curr_blurred = np.squeeze(curr_blurred)  # Ensure 2D

        # Ensure ratio computation works with 2D arrays
        if prev_blurred.ndim != 2 or curr_blurred.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays, got shapes {prev_blurred.shape}, {curr_blurred.shape}"
            )

        weighted_sum += (
            weights[i - 1]
            * prev_blurred
            / (curr_blurred + base_activation / scales[i] ** 2)
        )
        prev_blurred = curr_blurred  # Update for the next iteration

    return weighted_sum
