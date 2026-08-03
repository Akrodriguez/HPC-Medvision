
"""
cpu_vs_openmp.py
Phase 8: Rigorous Sequential vs OpenMP benchmarking (3 repetitions per config)
"""
import subprocess
import re
import csv
import os

OPENMP_DIR = os.path.expanduser("~/HPC-MedVision/openmp")
RESULTS_CSV = os.path.expanduser("~/HPC-MedVision/results/cpu_openmp_benchmark.csv")
REPETITIONS = 3
THREAD_COUNTS = [1, 2, 4, 8, 16, 32]

TIME_PATTERN = re.compile(r"Total execution time:\s*([\d.]+)\s*seconds")


def run_binary(binary, env=None):
    result = subprocess.run(
        [os.path.join(OPENMP_DIR, binary)],
        cwd=OPENMP_DIR,
        capture_output=True,
        text=True,
        env=env
    )
    match = TIME_PATTERN.search(result.stdout)
    if not match:
        print(f"WARNING: could not parse timing:\n{result.stdout}\n{result.stderr}")
        return None
    return float(match.group(1))


def main():
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    rows = []

    print("=== Sequential (preprocess_seq) ===")
    for rep in range(1, REPETITIONS + 1):
        t = run_binary("preprocess_seq")
        print(f"  Run {rep}: {t:.3f}s")
        rows.append({"config": "sequential", "threads": 1, "run": rep, "time_seconds": t})

    print("=== OpenMP (preprocess_omp) ===")
    for threads in THREAD_COUNTS:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        for rep in range(1, REPETITIONS + 1):
            t = run_binary("preprocess_omp", env=env)
            print(f"  Threads={threads} Run {rep}: {t:.3f}s")
            rows.append({"config": "openmp", "threads": threads, "run": rep, "time_seconds": t})

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "threads", "run", "time_seconds"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
