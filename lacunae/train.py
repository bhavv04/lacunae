import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from lacunae.dataset import SliceDataset
from lacunae.model import UNet
import matplotlib.pyplot as plt
from pathlib import Path


def train(
    data_dir: str = "data/singlecoil_test", 
    epochs: int = 30,   #changing this from 10 due to overfitting, also 10 barely procuded anything decent results
    batch_size: int = 4,
    lr: float = 1e-3,
    acceleration: int = 4,
    save_dir: str = "results",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # dataset
    dataset = SliceDataset(data_dir, acceleration=acceleration)
    print(f"Total slices: {len(dataset)}")

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    Path(save_dir).mkdir(exist_ok=True)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch["masked_kspace"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["masked_kspace"].to(device)
                targets = batch["target"].to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        # save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_dir}/unet_epoch{epoch+1}.pt")

    # plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Training curve")
    plt.legend()
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()
    print(f"Loss curve saved to {save_dir}/loss_curve.png")

    return model


if __name__ == "__main__":
    train()