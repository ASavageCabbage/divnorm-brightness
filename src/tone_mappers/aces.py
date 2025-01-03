import numpy as np


def aces(
    image: np.ndarray,
    a: float = 2.51,
    b: float = 0.03,
    c: float = 2.43,
    d: float = 0.59,
    e: float = 0.14,
    key: float | None = 0.18,
):
    """ACES global tone mapper with default values as per
    https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    """
    if key is not None:
        rms = np.sqrt(np.mean(image**2))
        image = (key / rms) * image
    return (image * (a * image + b)) / (image * (c * image + d) + e)
