import numpy as np

from src.tone_mappers.util import rgb_to_xyz, xyz_to_lxy, lxy_to_rgb


def linear(image: np.ndarray, pct: float = 1.0) -> np.ndarray:
    """Simple linear tone mapping where RGB values are scaled between (0, 1)
    
    The pct-percentile of RGB values is used as the maximum value.
    """
    return image / np.percentile(image, pct)


def linear_luminance(image: np.ndarray, pct: float = 100) -> np.ndarray:
    """Simple linear tone mapping where RGB values are scaled between (0, 1)
    
    The pct-percentile of luminance values is used for scaling.
    """
    X, Y, Z = rgb_to_xyz(image)
    L, x_chroma, y_chroma = xyz_to_lxy(X, Y, Z)
    L = L / np.percentile(L, pct)
    return lxy_to_rgb(L, x_chroma, y_chroma)
