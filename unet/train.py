
"""
train.py
HPC-MedVision - Phase 6: U-Net Training with Checkpointing

Designed to run across multiple short PBS sessions.
Automatically resumes from the last saved checkpoint if one exists.

Usage: python3 train.py [--epochs N]
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LGGDataset
from model import LightUNet

CHECKPOINT_DIR = os.path.expanduser("~/HPC-MedVision/results/checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
LOG_PATH = os.path.expanduser("~/HPC-MedVision/results/training_log.csv")

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEFAULT_TOTAL_EPOCHS = 20


def dice_loss(pred_logits, target, smooth=1e-6):
    """
    Combined BCE + Dice loss. BCE gives a strong per-pixel gradient signal
    that resists collapse; Dice directly optimizes overlap.
    pred_logits: raw model output (before sigmoid)
    target: binary ground truth mask
    """
    bce = nn.functional.binary_cross_entropy_with_logits(pred_logits, target)

    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    dice = 1.0 - dice_coeff.mean()

    return 0.5 * bce + 0.5 * dice


def dice_coefficient(pred_logits, target, smooth=1e-6, threshold=0.5):
    """Compute Dice coefficient (not loss) for monitoring, using a hard threshold."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, val_dices, is_best=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_dices": val_dices,
    }
    torch.save(checkpoint_data, CHECKPOINT_PATH)
    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")
        torch.save(checkpoint_data, best_path)


def load_checkpoint(model, optimizer, device):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        val_dices = checkpoint["val_dices"]
        print(f"Resumed from checkpoint at epoch {checkpoint['epoch']}. "
              f"Continuing from epoch {start_epoch}.")
        return start_epoch, train_losses, val_losses, val_dices
    else:
        print("No checkpoint found. Starting fresh from epoch 0.")
        return 0, [], [], []


def append_log(epoch, train_loss, val_loss, val_dice, epoch_time):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a") as f:
        if write_header:
            f.write("epoch,train_loss,val_loss,val_dice,epoch_time_sec\n")
        f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_dice:.6f},{epoch_time:.2f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_TOTAL_EPOCHS,
                        help="Total number of epochs to train to (not additional epochs)")
    args = parser.parse_args()
    total_epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Datasets and loaders ---
    train_ds = LGGDataset(split="train")
    val_ds = LGGDataset(split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model, optimizer ---
    model = LightUNet(in_channels=1, out_channels=1, base_filters=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Resume if checkpoint exists ---
    start_epoch, train_losses, val_losses, val_dices = load_checkpoint(model, optimizer, device)

    if start_epoch >= total_epochs:
        print(f"Already trained to epoch {start_epoch - 1}, target is {total_epochs - 1}. "
              f"Nothing to do. Increase --epochs to train further.")
        return

    print(f"Training from epoch {start_epoch} to {total_epochs - 1} "
          f"({total_epochs - start_epoch} epochs this run)")

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        # --- Training phase ---
        model.train()
        running_train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        train_loss = running_train_loss / len(train_ds)

        # --- Validation phase ---
        model.eval()
        running_val_loss = 0.0
        running_val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = dice_loss(outputs, masks)
                running_val_loss += loss.item() * images.size(0)
                running_val_dice += dice_coefficient(outputs, masks) * images.size(0)

        val_loss = running_val_loss / len(val_ds)
        val_dice = running_val_dice / len(val_ds)

        epoch_time = time.time() - epoch_start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
              f"Time: {epoch_time:.1f}s")

        append_log(epoch, train_loss, val_loss, val_dice, epoch_time)

        is_best = val_dice == max(val_dices)
        save_checkpoint(model, optimizer, epoch, train_losses, val_losses, val_dices, is_best=is_best)
        if is_best:
            print(f"  -> New best Val Dice: {val_dice:.4f} (saved to best_checkpoint.pth)")

    print("Training complete for this run.")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()

