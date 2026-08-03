
"""
dataset.py
HPC-MedVision - Phase 6: U-Net Dataset Loader

Loads MRI slice / tumor mask pairs for train/val/test splits,
based on the patient-level split files created in Phase 2.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

RAW_DIR = os.path.expanduser("~/HPC-MedVision/data/raw/kaggle_3m")
SPLIT_DIR = os.path.expanduser("~/HPC-MedVision/data")
IMG_SIZE = 128


def load_patient_list(split_name):
    """Read a patient ID list file (train/val/test_patients.txt)."""
    path = os.path.join(SPLIT_DIR, f"{split_name}_patients.txt")
    with open(path, "r") as f:
        patients = [line.strip() for line in f if line.strip()]
    return patients


def build_pairs(patients):
    """
    For each patient folder, find all MRI slice images and pair each
    with its corresponding mask file. Returns a list of (image_path, mask_path).
    """
    pairs = []
    for patient in patients:
        patient_dir = os.path.join(RAW_DIR, patient)
        if not os.path.isdir(patient_dir):
            continue

        files = os.listdir(patient_dir)
        image_files = [
            f for f in files
            if f.endswith(".tif") and "_mask" not in f
        ]

        for img_file in image_files:
            mask_file = img_file.replace(".tif", "_mask.tif")
            img_path = os.path.join(patient_dir, img_file)
            mask_path = os.path.join(patient_dir, mask_file)

            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))

    return pairs


class LGGDataset(Dataset):
    def __init__(self, split="train", img_size=IMG_SIZE):
        """
        split: one of "train", "val", "test"
        """
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.img_size = img_size
        self.split = split

        patients = load_patient_list(split)
        self.pairs = build_pairs(patients)

        print(f"[{split}] {len(patients)} patients, {len(self.pairs)} image/mask pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # --- Load image (grayscale) ---
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0  # normalize to [0,1]

        # --- Load mask (grayscale, then binarize) ---
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)  # binarize: tumor=1.0, background=0.0

        # --- Convert to PyTorch tensors, shape (1, H, W) ---
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return img_tensor, mask_tensor


if __name__ == "__main__":
    # Quick sanity check when run directly: python3 dataset.py
    for split in ["train", "val", "test"]:
        ds = LGGDataset(split=split)
        img, mask = ds[0]
        print(f"  Sample image shape: {img.shape}, dtype: {img.dtype}, "
              f"range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"  Sample mask shape:  {mask.shape}, dtype: {mask.dtype}, "
              f"unique values: {torch.unique(mask).tolist()}")

