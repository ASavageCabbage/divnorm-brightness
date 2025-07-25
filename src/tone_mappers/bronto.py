import numpy as np

from src.dino.bloom import apply_bloom
from src.dino.opencv import (
    arcmin2_to_pixel2,
    dinos_efficient,
    generate_mipmap,
    efficient_blur_from_mipmap,
)
from src.dino.util import generate_scales
from src.tone_mappers.util import rgb_to_xyz, xyz_to_lxy, lxy_to_rgb


def brontosaurus_rgb(
    rgb_image: np.ndarray,
    k: float = 0.18,
    white_base: float = 2.2,
    *args,
    **kwargs,
) -> np.ndarray:
    """Version of BRONTO using DINOS more directly as a "local average".

    White balance normalized, per-colour channel version.
    """
    # White balance normalization
    rgb_sum = np.sum(rgb_image, axis=(0, 1))
    rgb_sum /= np.sum(rgb_sum**2) ** (1 / 2)
    rgb_image /= rgb_sum
    tonemapped_bgr = []
    for channel in range(rgb_image.shape[-1]):
        L = rgb_image[:, :, channel]
        brightness_map = dinos_efficient(L, *args, **kwargs)
        local_white = white_base**brightness_map
        # Restore white balance
        tonemapped_L = (k / local_white) * L * rgb_sum[channel]
        tonemapped_bgr.append(tonemapped_L)
    return np.stack(tonemapped_bgr, axis=-1)


def brontosaurus(
    rgb_image: np.ndarray,
    k: float = 0.18,
    white_base: float = 2.2,
    *args,
    bloom: bool = True,
    image_fov_degrees: float = 72,
    **kwargs,
) -> np.ndarray:
    """Version of BRONTO using DINOS more directly as a "local average"."""
    X, Y, Z = rgb_to_xyz(rgb_image)
    L, x_chroma, y_chroma = xyz_to_lxy(X, Y, Z)
    if bloom:
        L_bloom = apply_bloom(L, image_fov_degrees=image_fov_degrees)
        brightness_map = dinos_efficient(L_bloom, *args, **kwargs)
    else:
        brightness_map = dinos_efficient(L, *args, **kwargs)
    local_white = white_base**brightness_map
    L_tonemapped = (k / local_white) * L
    return lxy_to_rgb(L_tonemapped, x_chroma, y_chroma)


def bronto_rgb(
    rgb_image: np.ndarray,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
    k: float = 0.3,
    w: float = 0.9,
    m: float = 1,
    d_nit_arcmin2: float = 100,
    image_fov_degrees: float = 72,
) -> np.ndarray:
    """BRightness Optimized Normalization Tone-mapping Operator.

    White balance normalized, per-colour channel version proposed by Thomas Roughton.
    """
    # White balance normalization
    rgb_sum = np.sum(rgb_image, axis=(0, 1))
    rgb_sum /= np.sum(rgb_sum**2) ** (1 / 2)
    rgb_image /= rgb_sum
    tonemapped_bgr = []
    for channel in range(rgb_image.shape[-1]):
        L = rgb_image[:, :, channel]
        width = L.shape[1]
        d = arcmin2_to_pixel2(d_nit_arcmin2, width, image_fov_degrees)

        mipmap = generate_mipmap(L, cs_ratio, num_scales)
        num_scales = len(mipmap)
        weights = [w**i for i in range(num_scales)]
        scales = generate_scales(width, cs_ratio, num_scales)
        center_response = efficient_blur_from_mipmap(mipmap, 0, cs_ratio)
        accum = np.zeros_like(L)
        c_sum = np.zeros_like(L)

        for i in range(1, num_scales):
            surround_response = efficient_blur_from_mipmap(mipmap, i, cs_ratio)
            w = weights[i - 1]
            _d = d / scales[i] ** 2
            c = w * np.abs((center_response + _d) / (surround_response + _d) - 1) + m
            accum += c * surround_response
            c_sum += c
            center_response = surround_response

        local_white = accum / c_sum
        # Restore white balance
        tonemapped_bgr.append((k / local_white) * L * rgb_sum[channel])
    return np.stack(tonemapped_bgr, axis=-1)


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
    width = L.shape[1]
    d = arcmin2_to_pixel2(d_nit_arcmin2, width, image_fov_degrees)

    mipmap = generate_mipmap(L, cs_ratio, num_scales)
    num_scales = len(mipmap)
    weights = [w**i for i in range(num_scales)]
    scales = generate_scales(width, cs_ratio, num_scales)
    center_response = efficient_blur_from_mipmap(mipmap, 0, cs_ratio)
    accum = np.zeros_like(L)
    c_sum = np.zeros_like(L)

    for i in range(1, num_scales):
        surround_response = efficient_blur_from_mipmap(mipmap, i, cs_ratio)
        w = weights[i - 1]
        _d = d / scales[i] ** 2
        c = w * np.abs((center_response + _d) / (surround_response + _d) - 1) + m
        accum += c * surround_response
        c_sum += c
        center_response = surround_response

    local_white = accum / c_sum
    L_tonemapped = (k / local_white) * L
    return lxy_to_rgb(L_tonemapped, x_chroma, y_chroma)
