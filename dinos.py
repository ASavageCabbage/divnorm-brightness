import cv2
import numpy as np


def _gaussian_blur(
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


def _generate_scales(width: int, cs_ratio: float, num_scales: int) -> list[float]:
    """Generate an exponential list of scales up to width * cs_ratio."""
    scale = width * cs_ratio
    scales = []
    for _ in range(num_scales):
        scales.insert(0, scale)
        scale /= cs_ratio
    return scales


def _rgb_to_xyz(image):
    # Conversion matrix from sRGB to XYZdef xyz_to_lxy(X, Y, Z):
    rgb_to_xyz_matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = np.dot(image, rgb_to_xyz_matrix.T)
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]  # X, Y, Z


def _xyz_to_lxy(X, Y, Z):
    denom = X + Y + Z + 1e-8  # Prevent division by zero
    x_chroma = X / denom
    y_chroma = Y / denom
    return Y, x_chroma, y_chroma  # Use Y as luminance


def _lxy_to_rgb(luminance, x_chroma, y_chroma):
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
    return rgb


def dinos(
    L: np.ndarray,
    gamma: float = 2.2,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
    w: float = 0.9,
    b: float = 1.0,
    d: float = 1.0,
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

    scales = _generate_scales(L.shape[1], cs_ratio, num_scales)
    weights = [w**i for i in range(len(scales))]

    # Initialize weighted_sum as a 2D array
    weighted_sum = np.zeros(L.shape[:2])

    # Compute ratios and weighted sum using only two blurred images at a time
    center_response = _gaussian_blur(L, scales[0])

    for i in range(1, len(scales)):
        surround_response = _gaussian_blur(L, scales[i])
        _b = b / scales[i] ** 2
        _d = d / scales[i] ** 2
        weighted_sum += weights[i - 1] * (
            (center_response + _b) / (surround_response + _d) - _b / _d
        )
        center_response = surround_response

    return weighted_sum


def bronto(
    rgb_image: np.ndarray,
    gamma: float = 2,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
    k: float = 0.25,
    w: float = 0.85,
    b: float = 1.0,
    d: float = 1.0,
):
    X, Y, Z = _rgb_to_xyz(rgb_image)
    luminance, x_chroma, y_chroma = _xyz_to_lxy(X, Y, Z)

    L = luminance ** (1.0 / gamma)
    scales = _generate_scales(L.shape[1], cs_ratio, num_scales)
    weights = [w**i for i in range(len(scales))]
    center_response = _gaussian_blur(L, scales[0])
    accum = np.zeros_like(luminance)
    response_sum = np.zeros_like(luminance)

    for i in range(1, len(scales)):
        surround_response = _gaussian_blur(L, scales[i])
        w = weights[i - 1]
        _b = b / scales[i] ** 2
        _d = w * d / scales[i] ** 2
        response = weights[i - 1] * (
            (center_response + _b) / (surround_response + _d) - _b / _d
        )
        response = np.abs(response)
        accum += response * (center_response**gamma)
        response_sum += response
        center_response = surround_response

    local_white = accum / response_sum
    tonemapped_luminance = (k / local_white) * luminance
    return _lxy_to_rgb(tonemapped_luminance, x_chroma, y_chroma)
