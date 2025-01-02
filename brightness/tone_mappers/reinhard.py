import numpy as np


#assumes input luminances are linear and exposure is already done
def reinhard_tone_map(
    L: np.ndarray, key: float = 0.18, L_white: float = 1.0
) -> np.ndarray:
    """Tone mapper described by Reinhard et al"""
    tone_mapped = L / (1 + L / (L_white**2)
    )
    return tone_mapped
