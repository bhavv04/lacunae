import numpy as np


def build_random_cartesian_mask(shape: tuple, acceleration: int, center_fractions: float = 0.08) -> np.ndarray:
    """
    Random Cartesian undersampling mask.
    Always keeps center low-frequency lines, randomly samples the rest.
    """
    num_cols = shape[-1]
    num_low_freqs = int(round(num_cols * center_fractions))
    num_keep = num_cols // acceleration

    mask = np.zeros(num_cols, dtype=np.float32)

    center_start = (num_cols - num_low_freqs) // 2
    mask[center_start: center_start + num_low_freqs] = 1

    remaining = num_keep - num_low_freqs
    candidates = list(set(range(num_cols)) - set(range(center_start, center_start + num_low_freqs)))
    chosen = np.random.choice(candidates, size=remaining, replace=False)
    mask[chosen] = 1

    return mask


def build_equispaced_mask(shape: tuple, acceleration: int, center_fractions: float = 0.08) -> np.ndarray:
    """
    Equispaced Cartesian mask.
    Samples every Nth column deterministically, always retains center.
    """
    num_cols = shape[-1]
    num_low_freqs = int(round(num_cols * center_fractions))

    mask = np.zeros(num_cols, dtype=np.float32)

    # always keep center
    center_start = (num_cols - num_low_freqs) // 2
    mask[center_start: center_start + num_low_freqs] = 1

    # equispaced over remaining columns
    for i in range(0, num_cols, acceleration):
        mask[i] = 1

    return mask


def build_radial_mask(shape: tuple, acceleration: int, center_fractions: float = 0.08) -> np.ndarray:
    """
    Radial-weighted mask.
    Samples columns with probability inversely proportional to distance
    from k-space center, simulating radial acquisition geometry.
    """
    num_cols = shape[-1]
    num_keep = num_cols // acceleration
    center = num_cols // 2

    weights = 1.0 / (np.abs(np.arange(num_cols) - center) + 1)
    weights /= weights.sum()

    chosen = np.random.choice(num_cols, size=num_keep, replace=False, p=weights)
    mask = np.zeros(num_cols, dtype=np.float32)
    mask[chosen] = 1

    return mask


def build_variable_density_mask(shape: tuple, acceleration: int, center_fractions: float = 0.08) -> np.ndarray:
    """
    Variable-density mask.
    Quadratic falloff from k-space center — denser low-frequency
    sampling, sparser high-frequency sampling.
    """
    num_cols = shape[-1]
    num_keep = num_cols // acceleration
    center = num_cols // 2

    distances = np.abs(np.arange(num_cols) - center).astype(np.float32)
    weights = 1.0 / (distances ** 2 + 1)
    weights /= weights.sum()

    chosen = np.random.choice(num_cols, size=num_keep, replace=False, p=weights)
    mask = np.zeros(num_cols, dtype=np.float32)
    mask[chosen] = 1