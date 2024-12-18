import numpy as np


def aces(
    L: np.ndarray,
    a: float = 2.51,
    b: float = 0.03,
    c: float = 2.43,
    d: float = 0.59,
    e: float = 0.14,
):
    """ACES global tone mapper with default values as per
    https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    """
    return (L * (a * L + b)) / (L * (c * L + d) + e)
