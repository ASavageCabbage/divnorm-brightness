import numpy as np

from src.dino.opencv import arcmin2_to_pixel2, gaussian_blur
from src.dino.util import generate_scales
from src.tone_mappers.util import rgb_to_xyz, xyz_to_lxy, lxy_to_rgb


def bronto(
    rgb_image: np.ndarray,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
    gamma: float = 1.0,
    k: float = 0.3,
    w: float = 0.9,
    m: float = 0.1,
    d_nit_arcmin2: float = 100,
    image_fov_degrees: float = 72,
) -> np.ndarray:
    """BRightness Optimized Normalization Tone-mapping Operator."""
    X, Y, Z = rgb_to_xyz(rgb_image)
    L, x_chroma, y_chroma = xyz_to_lxy(X, Y, Z)
    L_gamma = L ** (1 / gamma)
    width = L.shape[1]
    d = arcmin2_to_pixel2(d_nit_arcmin2, width, image_fov_degrees)

    scales = generate_scales(width, cs_ratio, num_scales)
    weights = [w**i for i in range(len(scales))]
    center_response_gamma = gaussian_blur(L_gamma, scales[0])
    accum = np.zeros_like(L)
    c_sum = np.zeros_like(L)

    for i in range(1, len(scales)):
        surround_response = gaussian_blur(L, scales[i])
        surround_response_gamma = surround_response if gamma == 1 else gaussian_blur(L_gamma, scales[i]) 
        w = weights[i - 1]
        _d = d / scales[i] ** 2
        c = w * np.abs((center_response_gamma + _d) / (surround_response_gamma + _d) - 1) + m
        accum += c * surround_response
        c_sum += c
        center_response_gamma = surround_response_gamma

    local_white = accum / c_sum
    L_tonemapped = (k / local_white) * L
    return lxy_to_rgb(L_tonemapped, x_chroma, y_chroma)
