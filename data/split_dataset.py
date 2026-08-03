"""
split_dataset.py
Splits LGG MRI patient folders into train/validation/test sets.
Split is done by PATIENT ID to avoid data leakage between sets.
"""

import os
import random

# ---- Configuration ----
RAW_DATA_DIR = os.path.expanduser("~/HPC-MedVision/data/raw/kaggle_3m")
OUTPUT_DIR = os.path.expanduser("~/HPC-MedVision/data")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

def main():
    all_entries = os.listdir(RAW_DATA_DIR)
    patients = sorted([
        entry for entry in all_entries
        if os.path.isdir(os.path.join(RAW_DATA_DIR, entry))
    ])

    print(f"Total patient folders found: {len(patients)}")

    random.seed(RANDOM_SEED)
    random.shuffle(patients)

    n_total = len(patients)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]

    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients:   {len(val_patients)}")
    print(f"Test patients:  {len(test_patients)}")

    def write_list(filename, patient_list):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            for p in patient_list:
                f.write(p + "\n")
        print(f"Wrote {len(patient_list)} entries to {path}")

    write_list("train_patients.txt", train_patients)
    write_list("val_patients.txt", val_patients)
    write_list("test_patients.txt", test_patients)

    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    assert len(train_set & val_set) == 0, "Overlap between train and val!"
    assert len(train_set & test_set) == 0, "Overlap between train and test!"
    assert len(val_set & test_set) == 0, "Overlap between val and test!"

    print("Sanity check passed: no patient overlap between splits.")

if __name__ == "__main__":
    main()
