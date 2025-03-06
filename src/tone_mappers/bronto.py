import numpy as np

from src.dino.opencv import gaussian_blur
from src.dino.util import generate_scales
from src.tone_mappers.util import rgb_to_xyz, xyz_to_lxy, lxy_to_rgb


def bronto(
    rgb_image: np.ndarray,
    gamma: float = 2.2,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
    k: float = 0.25,
    w: float = 0.9,
    b: float = 1.0,
    d: float = 1.0,
) -> np.ndarray:
    """Brightness Optimized Normalization Tone-mapping Operator."""
    X, Y, Z = rgb_to_xyz(rgb_image)
    luminance, x_chroma, y_chroma = xyz_to_lxy(X, Y, Z)

    L = luminance ** (1.0 / gamma)
    scales = generate_scales(L.shape[1], cs_ratio, num_scales)
    weights = [w**i for i in range(len(scales))]
    center_response = gaussian_blur(L, scales[0])
    accum = np.zeros_like(luminance)
    response_sum = np.zeros_like(luminance)

    for i in range(1, len(scales)):
        surround_response = gaussian_blur(L, scales[i])
        w = weights[i - 1]
        _b = b / scales[i] ** 2
        _d = w * d / scales[i] ** 2
        response = w * ((center_response + _b) / (surround_response + _d) - _b / _d)
        response = np.abs(response)
        accum += response * (center_response)
        response_sum += response
        center_response = surround_response

    local_white = accum / response_sum
    tonemapped_luminance = (k / local_white) * luminance
    return lxy_to_rgb(tonemapped_luminance, x_chroma, y_chroma)
