
"""
training_curves.py
Phase 9: Generate training loss and validation Dice curves from training_log.csv
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.expanduser("~/HPC-MedVision/results")
LOG_CSV = os.path.join(RESULTS_DIR, "training_log.csv")


def main():
    df = pd.read_csv(LOG_CSV)
    df = df.sort_values("epoch")

    # --- Loss curve: train vs val ---
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker='o', markersize=3, label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker='o', markersize=3, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCE + Dice)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    # --- Dice curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_dice"], marker='o', markersize=3, color="green", label="Validation Dice")
    best_epoch = df.loc[df["val_dice"].idxmax()]
    plt.scatter([best_epoch["epoch"]], [best_epoch["val_dice"]], color="red", zorder=5,
                label=f"Best: {best_epoch['val_dice']:.4f} (epoch {int(best_epoch['epoch'])})")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Score over Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "dice_curve.png"), dpi=150)
    plt.close()

    print("Saved loss_curve.png and dice_curve.png")
    print(f"Total epochs logged: {len(df)}")
    print(f"Final train loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"Final val loss:   {df['val_loss'].iloc[-1]:.4f}")
    print(f"Best val Dice:    {best_epoch['val_dice']:.4f} at epoch {int(best_epoch['epoch'])}")


if __name__ == "__main__":
    main()
