
/*
 * gaussian_blur.cu
 * HPC-MedVision - Phase 5: CUDA Gaussian Blur
 *
 * Applies a 5x5 Gaussian blur kernel to a single MRI image on the GPU.
 * Each CUDA thread computes exactly one output pixel.
 *
 * Usage: ./gaussian_blur <input_image_path> <output_image_path>
 */

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define CUDA_CHECK(call)                                                      \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__     \
                       << " -> " << cudaGetErrorString(err) << std::endl;    \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

// 5x5 Gaussian kernel (sigma ~1.0), normalized so weights sum to 1
__constant__ float d_gaussianKernel[25] = {
    1,  4,  7,  4, 1,
    4, 16, 26, 16, 4,
    7, 26, 41, 26, 7,
    4, 16, 26, 16, 4,
    1,  4,  7,  4, 1
};
const float GAUSSIAN_KERNEL_SUM = 273.0f;

// CUDA kernel: each thread computes one output pixel (grayscale)
__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output,
                                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int halfSize = 2; // 5x5 kernel -> radius 2

    for (int ky = -halfSize; ky <= halfSize; ky++) {
        for (int kx = -halfSize; kx <= halfSize; kx++) {
            int px = x + kx;
            int py = y + ky;

            // Clamp to image boundaries (edge replication)
            px = min(max(px, 0), width - 1);
            py = min(max(py, 0), height - 1);

            float weight = d_gaussianKernel[(ky + halfSize) * 5 + (kx + halfSize)];
            sum += weight * static_cast<float>(input[py * width + px]);
        }
    }

    output[y * width + x] = static_cast<unsigned char>(sum / GAUSSIAN_KERNEL_SUM);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];

    // --- Load image as grayscale ---
    cv::Mat img = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "ERROR: Could not load image " << inputPath << std::endl;
        return 1;
    }

    int width = img.cols;
    int height = img.rows;
    size_t imgSize = width * height * sizeof(unsigned char);

    std::cout << "Image loaded: " << width << "x" << height << std::endl;

    // --- Allocate GPU memory ---
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, imgSize));
    CUDA_CHECK(cudaMalloc(&d_output, imgSize));

    // --- Copy input image to GPU ---
    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_input, img.data, imgSize, cudaMemcpyHostToDevice));
    auto t1 = std::chrono::high_resolution_clock::now();

    // --- Configure grid/block dimensions ---
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // --- Launch kernel and time it ---
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t3 = std::chrono::high_resolution_clock::now();

    // --- Copy result back to CPU ---
    cv::Mat result(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(result.data, d_output, imgSize, cudaMemcpyDeviceToHost));
    auto t4 = std::chrono::high_resolution_clock::now();

    // --- Save output ---
    cv::imwrite(outputPath, result);

    // --- Report timings ---
    std::chrono::duration<double, std::milli> h2d = t1 - t0;
    std::chrono::duration<double, std::milli> kernel = t3 - t2;
    std::chrono::duration<double, std::milli> d2h = t4 - t3;

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Host->Device transfer: " << h2d.count() << " ms" << std::endl;
    std::cout << "Kernel execution:      " << kernel.count() << " ms" << std::endl;
    std::cout << "Device->Host transfer: " << d2h.count() << " ms" << std::endl;
    std::cout << "Total GPU pipeline:    " << (h2d + kernel + d2h).count() << " ms" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Output saved to: " << outputPath << std::endl;

    // --- Cleanup ---
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

