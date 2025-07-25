from typing import Callable

import numpy as np
from scipy.ndimage import convolve
from scipy.signal import fftconvolve


def photopic_psf(theta: np.ndarray) -> np.ndarray:
    """Point spread function simulating bloom in the human eye.

    From Equation (5) in
    Physically-Based Glare Effects for Digital Images (Spencer et al.)
    """
    f0 = 2.61e6 * np.exp(-((theta / 0.02) ** 2))
    f1 = 20.91 * (theta + 0.02) ** (-3)
    f2 = 72.37 * (theta + 0.02) ** (-2)
    return 0.384 * f0 + 0.478 * f1 + 0.138 * f2


def apply_bloom(
    L: np.ndarray,
    image_fov_degrees: float = 72,
    mean_contrast_threshold: float = 5,
    psf: Callable[[np.ndarray], np.ndarray] = photopic_psf,
    kernel_fov_degrees: float = 20,
) -> np.ndarray:
    """Apply bloom to a luminance map using a particular PSF."""
    width = L.shape[1]
    kernel_width = round(width * kernel_fov_degrees / image_fov_degrees)
    center = kernel_width / 2
    X, Y = np.indices((kernel_width, kernel_width))
    deg_from_center = (
        np.sqrt((X - center) ** 2 + (Y - center) ** 2) * image_fov_degrees / width
    )
    psf_kernel = psf(deg_from_center)
    # Normalize PSF so it integrates to 1 over the kernel
    psf_kernel /= np.sum(psf_kernel)

    to_bloom = np.zeros_like(L)
    over_threshold = L > (mean_contrast_threshold * np.mean(L))
    to_bloom[over_threshold] = L[over_threshold]
    try:
        bloomed = convolve(to_bloom, psf_kernel, mode="nearest")
    except MemoryError:
        print("Direct convolution ran out of memory, using FFT instead")
        bloomed = fftconvolve(to_bloom, psf_kernel, mode="same")

    output = np.copy(L)
    output[over_threshold] = 0
    output += bloomed
    return output
