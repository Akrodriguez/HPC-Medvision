
"""
evaluate.py
HPC-MedVision - Phase 6: Test-Set Evaluation

Loads the best available checkpoint and computes Dice, IoU, Precision, Recall,
and pixel accuracy on the held-out test set.
"""

import os
import torch
from torch.utils.data import DataLoader

from dataset import LGGDataset
from model import LightUNet

CHECKPOINT_PATH = os.path.expanduser("~/HPC-MedVision/results/checkpoints/latest_checkpoint.pth")
RESULTS_PATH = os.path.expanduser("~/HPC-MedVision/results/test_evaluation.txt")

BATCH_SIZE = 16
THRESHOLD = 0.5


def compute_metrics(pred_logits, target, smooth=1e-6):
    """
    Computes TP/FP/FN/TN counts for a batch.
    Returns raw counts (not ratios) so they can be correctly summed
    across the whole dataset before computing final metrics.
    """
    pred = (torch.sigmoid(pred_logits) > THRESHOLD).float()

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()

    return tp, fp, fn, tn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_ds = LGGDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = LightUNet(in_channels=1, out_channels=1, base_filters=32).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(Val Dice at save time: {checkpoint['val_dices'][checkpoint['epoch']]:.4f})")

    total_tp, total_fp, total_fn, total_tn = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            tp, fp, fn, tn = compute_metrics(outputs, masks)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    smooth = 1e-6
    dice = (2 * total_tp + smooth) / (2 * total_tp + total_fp + total_fn + smooth)
    iou = (total_tp + smooth) / (total_tp + total_fp + total_fn + smooth)
    precision = (total_tp + smooth) / (total_tp + total_fp + smooth)
    recall = (total_tp + smooth) / (total_tp + total_fn + smooth)
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)

    results = (
        f"=== HPC-MedVision Test Set Evaluation ===\n"
        f"Checkpoint epoch: {checkpoint['epoch']}\n"
        f"Test set size: {len(test_ds)} images\n"
        f"-----------------------------------\n"
        f"Dice Score:  {dice:.4f}\n"
        f"IoU:         {iou:.4f}\n"
        f"Precision:   {precision:.4f}\n"
        f"Recall:      {recall:.4f}\n"
        f"Accuracy:    {accuracy:.4f}\n"
        f"-----------------------------------\n"
        f"Total pixels evaluated: {int(total_tp + total_fp + total_fn + total_tn):,}\n"
        f"True Positive pixels:  {int(total_tp):,}\n"
        f"False Positive pixels: {int(total_fp):,}\n"
        f"False Negative pixels: {int(total_fn):,}\n"
        f"True Negative pixels:  {int(total_tn):,}\n"
    )

    print(results)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write(results)

    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
