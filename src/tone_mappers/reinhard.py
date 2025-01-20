import numpy as np


def reinhard(
    image: np.ndarray, key: float = 0.18, l_white: float = 1.0, gamma: float = 2.2
) -> np.ndarray:
    """Tone mapper described by Reinhard et al"""
    image = (key / (np.mean(image ** (1.0/gamma)) ** gamma)) * image
    return image / (1 + image / (l_white**2))
