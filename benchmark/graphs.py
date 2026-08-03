
"""
graphs.py
Phase 8: Generate benchmarking graphs from CSV results.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.expanduser("~/HPC-MedVision/results")
CPU_CSV = os.path.join(RESULTS_DIR, "cpu_openmp_benchmark.csv")
CUDA_CSV = os.path.join(RESULTS_DIR, "cuda_benchmark.csv")


def plot_cpu_openmp():
    df = pd.read_csv(CPU_CSV)
    seq_avg = df[df.config == "sequential"]["time_seconds"].mean()

    omp = df[df.config == "openmp"].groupby("threads")["time_seconds"].mean().reset_index()
    omp = omp.sort_values("threads")

    speedup = seq_avg / omp["time_seconds"]
    plt.figure(figsize=(7, 5))
    plt.plot(omp["threads"], speedup, marker='o', label="Measured speedup")
    plt.plot(omp["threads"], omp["threads"], linestyle="--", color="gray", label="Ideal linear speedup")
    plt.xlabel("Number of OpenMP threads")
    plt.ylabel("Speedup (relative to sequential)")
    plt.title("OpenMP Speedup vs Thread Count (avg of 3 runs)")
    plt.xscale("log", base=2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "speedup_chart.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    labels = ["Sequential"] + [f"{t} threads" for t in omp["threads"]]
    times = [seq_avg] + list(omp["time_seconds"])
    plt.bar(labels, times, color="steelblue")
    plt.ylabel("Execution time (seconds)")
    plt.title("Sequential vs OpenMP Runtime (avg of 3 runs)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "runtime_chart.png"), dpi=150)
    plt.close()

    print("Saved speedup_chart.png and runtime_chart.png")
    print(f"Sequential avg: {seq_avg:.2f}s")
    print(omp)


def plot_cuda():
    df = pd.read_csv(CUDA_CSV)
    warm = df[df.phase == "warm"].groupby("kernel")["kernel_time_ms"].mean()
    cold = df[df.phase == "cold"].groupby("kernel")["kernel_time_ms"].mean()

    kernels = warm.index.tolist()
    warm_vals = [warm[k] for k in kernels]
    cold_vals = [cold.get(k, 0) for k in kernels]

    x = range(len(kernels))
    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.2 for i in x], cold_vals, width=0.4, label="Cold (1st launch, incl. CUDA init)")
    plt.bar([i + 0.2 for i in x], warm_vals, width=0.4, label="Warm (avg steady-state)")
    plt.xticks(list(x), kernels)
    plt.ylabel("Kernel execution time (ms)")
    plt.yscale("log")
    plt.title("CUDA Kernel: Cold-Start vs Warm Execution Time (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cuda_warmup_chart.png"), dpi=150)
    plt.close()

    print("Saved cuda_warmup_chart.png")
    print("Cold (ms):\n", cold)
    print("Warm (ms):\n", warm)


if __name__ == "__main__":
    plot_cpu_openmp()
    plot_cuda()
