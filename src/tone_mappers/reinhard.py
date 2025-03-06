import numpy as np

from src.tone_mappers.util import rgb_to_xyz, xyz_to_lxy, lxy_to_rgb


def reinhard(
    image: np.ndarray, key: float = 0.18, l_white: float = 1.0, gamma: float = 2.2
) -> np.ndarray:
    """Tone mapper described by Reinhard et al, using direct RGB values"""
    image = (key / (np.mean(image ** (1.0 / gamma)) ** gamma)) * image
    return image / (1 + image / (l_white**2))


def reinhard_luminance(
    image: np.ndarray, key: float = 0.18, l_white: float = 1.0, gamma: float = 2.2
) -> np.ndarray:
    """Tone mapper described by Reinhard et al, using LXY mapping"""
    X, Y, Z = rgb_to_xyz(image)
    L, x_chroma, y_chroma = xyz_to_lxy(X, Y, Z)
    L = (key / (np.mean(L ** (1.0 / gamma)) ** gamma)) * L
    L = L / (1 + L / (l_white**2))
    return lxy_to_rgb(L, x_chroma, y_chroma)
