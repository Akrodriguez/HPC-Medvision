
/*
 * sobel.cu
 * HPC-MedVision - Phase 5: CUDA Sobel Edge Detection
 *
 * Computes edge magnitude using the Sobel operator (Gx and Gy 3x3 kernels).
 * Each CUDA thread computes one output pixel.
 *
 * Usage: ./sobel <input_image_path> <output_image_path>
 */

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>

#define CUDA_CHECK(call)                                                      \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__     \
                       << " -> " << cudaGetErrorString(err) << std::endl;    \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

// Sobel Gx and Gy 3x3 kernels
__constant__ int d_sobelGx[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

__constant__ int d_sobelGy[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

__global__ void sobelKernel(const unsigned char* input, unsigned char* output,
                             int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float gx = 0.0f;
    float gy = 0.0f;
    int halfSize = 1; // 3x3 kernel -> radius 1

    for (int ky = -halfSize; ky <= halfSize; ky++) {
        for (int kx = -halfSize; kx <= halfSize; kx++) {
            int px = x + kx;
            int py = y + ky;

            // Clamp to image boundaries (edge replication)
            px = min(max(px, 0), width - 1);
            py = min(max(py, 0), height - 1);

            float pixelVal = static_cast<float>(input[py * width + px]);
            int kernelIdx = (ky + halfSize) * 3 + (kx + halfSize);

            gx += d_sobelGx[kernelIdx] * pixelVal;
            gy += d_sobelGy[kernelIdx] * pixelVal;
        }
    }

    float magnitude = sqrtf(gx * gx + gy * gy);
    // Clamp to valid pixel range [0, 255]
    magnitude = fminf(fmaxf(magnitude, 0.0f), 255.0f);

    output[y * width + x] = static_cast<unsigned char>(magnitude);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];

    cv::Mat img = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "ERROR: Could not load image " << inputPath << std::endl;
        return 1;
    }

    int width = img.cols;
    int height = img.rows;
    size_t imgSize = width * height * sizeof(unsigned char);

    std::cout << "Image loaded: " << width << "x" << height << std::endl;

    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, imgSize));
    CUDA_CHECK(cudaMalloc(&d_output, imgSize));

    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_input, img.data, imgSize, cudaMemcpyHostToDevice));
    auto t1 = std::chrono::high_resolution_clock::now();

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t3 = std::chrono::high_resolution_clock::now();

    cv::Mat result(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(result.data, d_output, imgSize, cudaMemcpyDeviceToHost));
    auto t4 = std::chrono::high_resolution_clock::now();

    cv::imwrite(outputPath, result);

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

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
