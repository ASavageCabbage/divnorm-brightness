import os

from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_SCALES = [2**i for i in range(10)]
DEFAULT_CENTER_SURROUND_RATIO = np.log(2)
DEFAULT_TRANSITION_FLUX = 0.0
DEFAULT_EXPONENTIAL_WEIGHT_DECAY = 0.0

DEFAULT_CMAP = "gnuplot2"


def perceived_brightness(
    pixels: np.ndarray,
    scales: list[float] = DEFAULT_SCALES,
    center_surround_ratio: float = DEFAULT_CENTER_SURROUND_RATIO,
    transition_flux: float = DEFAULT_TRANSITION_FLUX,
    exponential_weight_decay: float = DEFAULT_EXPONENTIAL_WEIGHT_DECAY,
) -> np.ndarray:
    """Applies Blommaert-Martens brightness perception model to input image."""

    # Input must be rectangular array of greyscale pixels in the range [0, 1].
    assert len(pixels.shape) == 2
    assert np.min(pixels) >= 0
    assert np.max(pixels) <= 1

    brightness = np.zeros(pixels.shape)
    for s in tqdm(scales, desc=f"Applying center-surround profiles at scales {scales}"):
        center = gaussian_filter(pixels, s)
        surround = gaussian_filter(pixels, center_surround_ratio * s)
        weight = np.exp(-exponential_weight_decay * np.log(s))
        brightness += weight * (center - surround) / (transition_flux / s**2 + center)

    return brightness


def plot_brightness(
    image_path: str, output_path: str = "", cmap: str = DEFAULT_CMAP, **kwargs
):
    """Plot the perceived brightness of the input image"""
    greyscale_image = Image.open(image_path).convert("L")
    pixels = np.asarray(greyscale_image) / 255
    brightness = perceived_brightness(pixels, **kwargs)
    ax = plt.imshow(brightness)
    plt.set_cmap(plt.get_cmap(cmap))
    plt.colorbar(ax)
    if output_path == "":
        output_path = f"{os.path.splitext(image_path)[0]}-brightness.png"
    plt.savefig(output_path)
    print("Output saved to", output_path)
    plt.close()
