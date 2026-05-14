import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from lacunae.transforms import apply_mask, build_mask

TARGET_SHAPE = (640, 372)

class SliceDataset(Dataset):
    def __init__(self, root_dir: str, acceleration: int = 4, center_fractions: float = 0.08):
        self.files = sorted(Path(root_dir).glob("*.h5"))
        self.acceleration = acceleration
        self.center_fractions = center_fractions
        self.slices = self._index_slices()

    def _index_slices(self):
        slices = []
        for filepath in self.files:
            with h5py.File(filepath, "r") as f:
                num_slices = f["kspace"].shape[0]
                for i in range(num_slices):
                    slices.append((filepath, i))
        return slices

    def __len__(self):
        return len(self.slices)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Pad image to TARGET_SHAPE if smaller, crop if larger."""
        h, w = image.shape
        th, tw = TARGET_SHAPE
        result = np.zeros(TARGET_SHAPE, dtype=np.float32)
        h_end = min(h, th)
        w_end = min(w, tw)
        result[:h_end, :w_end] = image[:h_end, :w_end]
        return result

    def __getitem__(self, idx):
        filepath, slice_idx = self.slices[idx]

        with h5py.File(filepath, "r") as f:
            kspace = f["kspace"][slice_idx]

        kspace = kspace.astype(np.complex64)

        # ground truth
        target = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))
        target = target / target.max()

        # build mask matching actual kspace width, then apply
        mask = build_mask(kspace.shape, self.acceleration, self.center_fractions)
        masked_kspace = apply_mask(kspace, mask)

        # naive IFFT reconstruction
        undersampled = np.abs(np.fft.ifft2(np.fft.ifftshift(masked_kspace)))
        undersampled = undersampled / (undersampled.max() + 1e-8)

        # pad both to fixed size after masking
        target = self._resize(target)
        undersampled = self._resize(undersampled)

        return {
            "masked_kspace": torch.from_numpy(undersampled).unsqueeze(0).float(),
            "target": torch.from_numpy(target).unsqueeze(0).float(),
        }


if __name__ == "__main__":
    dataset = SliceDataset("data/singlecoil_test")
    print(f"Total slices: {len(dataset)}")
    sample = dataset[0]
    print(f"Input shape:  {sample['masked_kspace'].shape}")
    print(f"Target shape: {sample['target'].shape}")