import numpy as np


def build_mask(shape: tuple, acceleration: int, center_fractions: float) -> np.ndarray:
    """
    Random Cartesian undersampling mask.
    Always keeps the center low-frequency lines.
    Randomly samples the rest to hit the target acceleration.
    """
    num_cols = shape[-1]
    num_low_freqs = int(round(num_cols * center_fractions))

    # how many total lines to keep
    num_keep = num_cols // acceleration

    # build mask
    mask = np.zeros(num_cols, dtype=np.float32)

    # always keep center
    center_start = (num_cols - num_low_freqs) // 2
    mask[center_start: center_start + num_low_freqs] = 1

    # randomly sample remaining lines
    remaining = num_keep - num_low_freqs
    candidates = list(set(range(num_cols)) - set(range(center_start, center_start + num_low_freqs)))
    chosen = np.random.choice(candidates, size=remaining, replace=False)
    mask[chosen] = 1

    return mask


def apply_mask(kspace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out unsampled k-space columns."""
    return kspace * mask