
"""
segmentation_demo.py
Phase 9: Visual demo - Original / Ground Truth / Predicted Mask side-by-side
for a handful of tumor-positive test-set samples.
"""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import LGGDataset
from model import LightUNet

CHECKPOINT_PATH = os.path.expanduser("~/HPC-MedVision/results/checkpoints/latest_checkpoint.pth")
OUTPUT_PATH = os.path.expanduser("~/HPC-MedVision/results/segmentation_demo.png")
NUM_SAMPLES = 4
THRESHOLD = 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_ds = LGGDataset(split="test")

    model = LightUNet(in_channels=1, out_channels=1, base_filters=32).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Find tumor-positive samples
    tumor_indices = []
    for i in range(len(test_ds)):
        _, mask = test_ds[i]
        if mask.max() > 0:
            tumor_indices.append(i)
        if len(tumor_indices) >= NUM_SAMPLES:
            break

    print(f"Found {len(tumor_indices)} tumor-positive samples for demo")

    fig, axes = plt.subplots(len(tumor_indices), 3, figsize=(9, 3 * len(tumor_indices)))
    if len(tumor_indices) == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row, idx in enumerate(tumor_indices):
            img, mask = test_ds[idx]
            img_batch = img.unsqueeze(0).to(device)
            output = model(img_batch)
            pred = (torch.sigmoid(output) > THRESHOLD).float().cpu().squeeze()

            img_np = img.squeeze().numpy()
            mask_np = mask.squeeze().numpy()
            pred_np = pred.numpy()

            axes[row, 0].imshow(img_np, cmap="gray")
            axes[row, 0].set_title("Original MRI" if row == 0 else "")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(img_np, cmap="gray")
            axes[row, 1].imshow(mask_np, cmap="Reds", alpha=0.5)
            axes[row, 1].set_title("Ground Truth" if row == 0 else "")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(img_np, cmap="gray")
            axes[row, 2].imshow(pred_np, cmap="Reds", alpha=0.5)
            dice = (2 * (pred_np * mask_np).sum()) / (pred_np.sum() + mask_np.sum() + 1e-6)
            axes[row, 2].set_title(f"Predicted (Dice: {dice:.3f})" if row == 0 else f"Dice: {dice:.3f}")
            axes[row, 2].axis("off")

    plt.suptitle("HPC-MedVision: Brain Tumor Segmentation Results (Test Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
