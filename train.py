import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import SliceDataset
from model import UNet
import matplotlib.pyplot as plt
from pathlib import Path
from torchmetrics.image import StructuralSimilarityIndexMeasure


def train(
    data_dir: str = "data/singlecoil_test",
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-3,
    acceleration: int = 8,
    save_dir: str = "results",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SliceDataset(data_dir, acceleration=acceleration)
    print(f"Total slices: {len(dataset)}")

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    Path(save_dir).mkdir(exist_ok=True)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch["masked_kspace"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = l1_loss(outputs, targets) + (1 - ssim_metric(outputs, targets))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["masked_kspace"].to(device)
                targets = batch["target"].to(device)
                outputs = model(inputs)
                val_loss += (l1_loss(outputs, targets) + (1 - ssim_metric(outputs, targets))).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_dir}/unet_8x_ssim_epoch{epoch+1}.pt")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("L1 + SSIM Loss")
    plt.title("Training curve (8x, SSIM loss)")
    plt.legend()
    plt.savefig(f"{save_dir}/loss_curve_8x_ssim.png")
    plt.close()
    print(f"Loss curve saved to results/loss_curve_8x_ssim.png")

    return model


if __name__ == "__main__":
    train()