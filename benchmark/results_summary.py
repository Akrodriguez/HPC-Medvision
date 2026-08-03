
"""
results_summary.py
Phase 9: Consolidate all benchmarking and evaluation results into one summary CSV.
"""
import os
import csv
import pandas as pd

RESULTS_DIR = os.path.expanduser("~/HPC-MedVision/results")
CPU_CSV = os.path.join(RESULTS_DIR, "cpu_openmp_benchmark.csv")
CUDA_CSV = os.path.join(RESULTS_DIR, "cuda_benchmark.csv")
TEST_EVAL_TXT = os.path.join(RESULTS_DIR, "test_evaluation.txt")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "results_summary.csv")


def parse_test_eval(path):
    metrics = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            for key in ["Dice Score", "IoU", "Precision", "Recall", "Accuracy"]:
                if line.startswith(key):
                    value = line.split(":")[-1].strip()
                    metrics[key] = value
    return metrics


def main():
    rows = []

    # --- CPU / OpenMP results ---
    cpu_df = pd.read_csv(CPU_CSV)
    seq_avg = cpu_df[cpu_df.config == "sequential"]["time_seconds"].mean()
    rows.append(["Preprocessing", "Sequential (CPU, 1 core)", "-", f"{seq_avg:.3f}", "seconds (avg of 3 runs, 3929 images)"])

    omp_avg = cpu_df[cpu_df.config == "openmp"].groupby("threads")["time_seconds"].mean()
    for threads, t in omp_avg.items():
        speedup = seq_avg / t
        rows.append(["Preprocessing", f"OpenMP ({threads} threads)", f"{speedup:.2f}x",
                     f"{t:.3f}", "seconds (avg of 3 runs, 3929 images)"])

    # --- CUDA kernel results ---
    cuda_df = pd.read_csv(CUDA_CSV)
    warm_avg = cuda_df[cuda_df.phase == "warm"].groupby("kernel")["kernel_time_ms"].mean()
    cold_avg = cuda_df[cuda_df.phase == "cold"].groupby("kernel")["kernel_time_ms"].mean()
    for kernel in warm_avg.index:
        rows.append(["CUDA Kernel", f"{kernel} (cold/1st launch)", "-",
                     f"{cold_avg[kernel]:.4f}", "ms (includes CUDA init)"])
        rows.append(["CUDA Kernel", f"{kernel} (warm/steady-state)", "-",
                     f"{warm_avg[kernel]:.4f}", "ms (avg of 19 images)"])

    # --- U-Net evaluation ---
    if os.path.exists(TEST_EVAL_TXT):
        metrics = parse_test_eval(TEST_EVAL_TXT)
        for name, value in metrics.items():
            rows.append(["U-Net Evaluation", name, "-", value, "test set, 589 images, 17 patients"])
    else:
        print(f"WARNING: {TEST_EVAL_TXT} not found, skipping U-Net metrics")

    # --- Write CSV ---
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Configuration", "Speedup", "Value", "Notes"])
        writer.writerows(rows)

    print(f"Saved {OUTPUT_CSV}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
