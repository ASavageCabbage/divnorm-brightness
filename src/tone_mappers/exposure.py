import numpy as np

from src.tone_mappers.util import rgb_to_xyz, xyz_to_lxy, lxy_to_rgb


def exposure(image: np.ndarray, key: float = 0.18, gamma: float = 2.2) -> np.ndarray:
    """Linear, exposure-based tone mapper L = key / avg(L_w) * L_w
    
    Uses direct RGB values in place of luminance.
    """
    return (key / (np.mean(image ** (1.0 / gamma)) ** gamma)) * image


def exposure_luminance(image: np.ndarray, key: float = 0.18, gamma: float = 2.2) -> np.ndarray:
    """Linear, exposure-based tone mapper L = key / avg(L_w) * L_w
    
    Maps RGB to luminance before scaling.
    """
    X, Y, Z = rgb_to_xyz(image)
    L, x_chroma, y_chroma = xyz_to_lxy(X, Y, Z)
    L = (key / (np.mean(L ** (1.0 / gamma)) ** gamma)) * L
    return lxy_to_rgb(L, x_chroma, y_chroma)
