import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader
from lacunae.dataset import SliceDataset
from lacunae.model import UNet
from pathlib import Path


def evaluate(
    data_dir: str = "data/singlecoil_test",
    checkpoint: str = "results/unet_epoch30.pt",
    save_dir: str = "results",
    num_visuals: int = 5,
    acceleration: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    # dataset
    dataset = SliceDataset(data_dir, acceleration=acceleration)
    
    # skip to more informative slices
    start = len(dataset) // 2
    indices = list(range(start, start + num_visuals))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    Path(save_dir).mkdir(exist_ok=True)

    all_ssim_input, all_ssim_recon = [], []
    all_psnr_input, all_psnr_recon = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["masked_kspace"].to(device)
            targets = batch["target"].to(device)

            outputs = model(inputs)

            # to numpy
            inp = inputs[0, 0].cpu().numpy()
            out = outputs[0, 0].cpu().numpy()
            tgt = targets[0, 0].cpu().numpy()

            # normalize to [0, 1] for metrics
            tgt_norm = (tgt - tgt.min()) / (tgt.max() - tgt.min() + 1e-8)
            out_norm = (out - out.min()) / (out.max() - out.min() + 1e-8)
            inp_norm = (inp - inp.min()) / (inp.max() - inp.min() + 1e-8)

            data_range = 1.0

            s_input = ssim(tgt_norm, inp_norm, data_range=data_range)
            s_recon = ssim(tgt_norm, out_norm, data_range=data_range)
            p_input = psnr(tgt_norm, inp_norm, data_range=data_range)
            p_recon = psnr(tgt_norm, out_norm, data_range=data_range)

            all_ssim_input.append(s_input)
            all_ssim_recon.append(s_recon)
            all_psnr_input.append(p_input)
            all_psnr_recon.append(p_recon)

            # save visuals for first num_visuals slices
            if i < num_visuals:
                _save_visual(inp_norm, out_norm, tgt_norm, s_input, s_recon,
                             p_input, p_recon, i, save_dir)

    # summary
    print("\n--- Evaluation Summary ---")
    print(f"{'Metric':<20} {'Undersampled':<20} {'U-Net Recon':<20}")
    print(f"{'SSIM':<20} {np.mean(all_ssim_input):.4f}               {np.mean(all_ssim_recon):.4f}")
    print(f"{'PSNR (dB)':<20} {np.mean(all_psnr_input):.2f}                {np.mean(all_psnr_recon):.2f}")
    print(f"\nVisuals saved to {save_dir}/")


def _save_visual(inp, out, tgt, s_inp, s_out, p_inp, p_out, idx, save_dir):
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.05)

    titles = [
        f"Undersampled IFFT\nSSIM: {s_inp:.4f}  PSNR: {p_inp:.2f}dB",
        f"U-Net Reconstruction\nSSIM: {s_out:.4f}  PSNR: {p_out:.2f}dB",
        "Ground Truth",
    ]
    images = [inp, out, tgt]

    for j, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[j])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.savefig(f"{save_dir}/visual_{idx:03d}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved visual_{idx:03d}.png")


if __name__ == "__main__":
    evaluate()