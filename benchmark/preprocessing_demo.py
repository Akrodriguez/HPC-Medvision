
"""
preprocessing_demo.py
Phase 9: Visual demo - Original / Gaussian Blur / Sobel / Histogram Equalization
side-by-side, using our compiled CUDA kernel binaries.
"""
import os
import subprocess
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUDA_DIR = os.path.expanduser("~/HPC-MedVision/cuda")
DATA_DIR = os.path.expanduser("~/HPC-MedVision/data/raw/kaggle_3m")
RESULTS_DIR = os.path.expanduser("~/HPC-MedVision/results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "preprocessing_demo.png")


def find_sample_image():
    for root, dirs, files in os.walk(DATA_DIR):
        for f in sorted(files):
            if f.endswith(".tif") and "_mask" not in f:
                return os.path.join(root, f)
    raise FileNotFoundError("No sample MRI image found")


def run_kernel(binary_name, input_path, output_path):
    binary_path = os.path.join(CUDA_DIR, binary_name)
    result = subprocess.run(
        [binary_path, input_path, output_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR running {binary_name}:\n{result.stdout}\n{result.stderr}")
        raise RuntimeError(f"{binary_name} failed")


def main():
    sample_image = find_sample_image()
    print(f"Using sample image: {sample_image}")

    blur_out = os.path.join(RESULTS_DIR, "demo_blur.tif")
    sobel_out = os.path.join(RESULTS_DIR, "demo_sobel.tif")
    histeq_out = os.path.join(RESULTS_DIR, "demo_histeq.tif")

    print("Running Gaussian Blur...")
    run_kernel("gaussian_blur", sample_image, blur_out)

    print("Running Sobel Edge Detection...")
    run_kernel("sobel", sample_image, sobel_out)

    print("Running Histogram Equalization...")
    run_kernel("histogram_equalization", sample_image, histeq_out)

    original = cv2.imread(sample_image, cv2.IMREAD_GRAYSCALE)
    blur = cv2.imread(blur_out, cv2.IMREAD_GRAYSCALE)
    sobel = cv2.imread(sobel_out, cv2.IMREAD_GRAYSCALE)
    histeq = cv2.imread(histeq_out, cv2.IMREAD_GRAYSCALE)

    images = [original, blur, sobel, histeq]
    titles = ["Original MRI", "Gaussian Blur", "Sobel Edge Detection", "Histogram Equalization"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.suptitle("HPC-MedVision: CUDA Image Enhancement Kernels", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Saved {OUTPUT_PATH}")

    # Cleanup intermediate files
    for f in [blur_out, sobel_out, histeq_out]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
