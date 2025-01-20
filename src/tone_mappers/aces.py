import numpy as np


def aces(
    image: np.ndarray,
    key: float | None = 0.18,
    a: float = 2.51,
    b: float = 0.03,
    c: float = 2.43,
    d: float = 0.59,
    e: float = 0.14,
):
    """ACES global tone mapper with default values as per
    https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    """
    if key is not None:
        smr = np.mean(np.sqrt(image)) ** 2
        image = (key / smr) * image
    return (image * (a * image + b)) / (image * (c * image + d) + e)
