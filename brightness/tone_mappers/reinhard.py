import numpy as np


def reinhard_tone_map(
    L: np.ndarray, key: float = 0.18, L_white: float = 1.0
) -> np.ndarray:
    """Tone mapper described by Reinhard et al"""
    gamma = 2.2
    # Reinhard takes the log average and this is akin to taking the average of gamma corrected values and linearizing it
    average = np.mean(L)
    linearized_average = np.power(average, gamma)
    linearized_luminances = np.power(L, gamma)
    scaled_linear_luminances = (key / linearized_average) * linearized_luminances
    tone_mapped = scaled_linear_luminances / (
        1 + scaled_linear_luminances / (L_white**2)
    )
    # gamma_corrected_tone_mapped = np.power(tone_mapped, 1.0/gamma)
    return tone_mapped
