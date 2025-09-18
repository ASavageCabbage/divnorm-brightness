import numpy as np

from src.dino.opencv import arcmin2_to_pixel2, gaussian_blur, generate_scales


def dn_brightness_model(
    L: np.ndarray,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
    w: float = 0.9,
    d_nit_arcmin2: float = 100,
    image_fov_degrees: float = 72,
) -> np.ndarray:
    """Apply DINOS model to array of linear absolute luminances.

    NOTE: The current scale is the center stdev at the current scale,
          the next scale up is the surround stdev at the current scale.
    """
    width = L.shape[1]
    d = arcmin2_to_pixel2(d_nit_arcmin2, width, image_fov_degrees)
    scales = generate_scales(width, cs_ratio, num_scales)
    weights = [w**i for i in range(len(scales))]
    # Initialize weighted_sum as a 2D array
    weighted_sum = np.zeros(L.shape[:2])
    # Compute ratios and weighted sum using only two blurred images at a time
    center_response = gaussian_blur(L, scales[0])

    for i in range(1, len(scales)):
        surround_response = gaussian_blur(L, scales[i])

        assert center_response.ndim == 2
        assert surround_response.ndim == 2

        _d = d / scales[i] ** 2
        weighted_sum += weights[i - 1] * (
            (center_response + _d) / (surround_response + _d) - 1
        )
        center_response = surround_response

    return weighted_sum


def blakeslee97_brightness_model(
    L: np.ndarray,
    cs_ratio: float = 2.0,
    w: float = 0.9,
    center_base_fov_degrees: float = 3,
    image_fov_degrees: float = 72,
) -> np.ndarray:
    """
    Apply Gaussian center-surround brightness model by Blakeslee
    to array of linear absolute luminances.

    NOTE: The current scale is the center stdev at the current scale,
          the next scale up is the surround stdev at the current scale.
    """
    width = L.shape[1]
    base_scale = width * center_base_fov_degrees / image_fov_degrees / 2
    scales = [
        base_scale / cs_ratio**2,
        base_scale / cs_ratio,
        base_scale,
        base_scale * cs_ratio,
        base_scale * cs_ratio**2,
        base_scale * cs_ratio**3,
        base_scale * cs_ratio**4,
    ]
    weights = [w**i for i in range(len(scales))]
    # Initialize weighted_sum as a 2D array
    weighted_sum = np.zeros(L.shape[:2])
    # Compute ratios and weighted sum using only two blurred images at a time
    center_response = gaussian_blur(L, scales[0])

    for i in range(1, len(scales)):
        surround_response = gaussian_blur(L, scales[i])

        assert center_response.ndim == 2
        assert surround_response.ndim == 2

        weighted_sum += weights[i - 1] * (center_response - surround_response)
        center_response = surround_response

    return weighted_sum


def blommaert_brightness_model(
    L: np.ndarray,
    scale_ratio: float = 2.0,
    num_scales: int = 13,
    cs_ratio: float = 1 / np.log(2),
    a: float = 0.36,
    d_nit_arcmin2: float = 100,
    image_fov_degrees: float = 72,
) -> np.ndarray:
    """
    Apply Gaussian brightness model by Blommaert and Martens
    to array of linear absolute luminances.
    """
    width = L.shape[1]
    d = arcmin2_to_pixel2(d_nit_arcmin2, width, image_fov_degrees)
    scales = generate_scales(width, scale_ratio, num_scales)
    weighted_sum = np.zeros(L.shape[:2])

    for i in range(0, len(scales) - 1):
        center_response = gaussian_blur(L, scales[i])
        surround_response = gaussian_blur(L, scales[i] * cs_ratio)

        assert center_response.ndim == 2
        assert surround_response.ndim == 2

        weighted_sum += (
            np.exp(-a * np.log(scales[i]))
            * (center_response - surround_response)
            / (center_response + d / scales[i] ** 2)
        )
        center_response = surround_response

    return weighted_sum
