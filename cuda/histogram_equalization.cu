
/*
 * histogram_equalization.cu
 * HPC-MedVision - Phase 5: CUDA Histogram Equalization
 *
 * Three-stage pipeline:
 *   1. GPU: build 256-bin histogram using atomicAdd (race-condition-safe)
 *   2. CPU: compute cumulative distribution function (CDF) from histogram
 *   3. GPU: remap each pixel using the CDF lookup table
 *
 * Usage: ./histogram_equalization <input_image_path> <output_image_path>
 */

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__     \
                       << " -> " << cudaGetErrorString(err) << std::endl;    \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

#define HIST_SIZE 256

__global__ void histogramKernel(const unsigned char* input, int* histogram,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    unsigned char pixelVal = input[y * width + x];
    atomicAdd(&histogram[pixelVal], 1);
}

__global__ void equalizeKernel(const unsigned char* input, unsigned char* output,
                                const unsigned char* lookupTable, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    unsigned char pixelVal = input[y * width + x];
    output[y * width + x] = lookupTable[pixelVal];
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
    int totalPixels = width * height;

    std::cout << "Image loaded: " << width << "x" << height << std::endl;

    unsigned char *d_input, *d_output, *d_lookupTable;
    int *d_histogram;

    CUDA_CHECK(cudaMalloc(&d_input, imgSize));
    CUDA_CHECK(cudaMalloc(&d_output, imgSize));
    CUDA_CHECK(cudaMalloc(&d_lookupTable, HIST_SIZE * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_histogram, HIST_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, HIST_SIZE * sizeof(int)));

    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_input, img.data, imgSize, cudaMemcpyHostToDevice));
    auto t1 = std::chrono::high_resolution_clock::now();

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    histogramKernel<<<gridSize, blockSize>>>(d_input, d_histogram, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t3 = std::chrono::high_resolution_clock::now();

    int histogram[HIST_SIZE];
    CUDA_CHECK(cudaMemcpy(histogram, d_histogram, HIST_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    long cdf[HIST_SIZE];
    cdf[0] = histogram[0];
    for (int i = 1; i < HIST_SIZE; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    long cdfMin = 0;
    for (int i = 0; i < HIST_SIZE; i++) {
        if (cdf[i] != 0) {
            cdfMin = cdf[i];
            break;
        }
    }

    unsigned char lookupTable[HIST_SIZE];
    for (int i = 0; i < HIST_SIZE; i++) {
        if (totalPixels - cdfMin == 0) {
            lookupTable[i] = 0;
        } else {
            float normalized = (static_cast<float>(cdf[i] - cdfMin) /
                                 static_cast<float>(totalPixels - cdfMin)) * 255.0f;
            normalized = fminf(fmaxf(normalized, 0.0f), 255.0f);
            lookupTable[i] = static_cast<unsigned char>(normalized);
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(d_lookupTable, lookupTable, HIST_SIZE * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    auto t5 = std::chrono::high_resolution_clock::now();

    equalizeKernel<<<gridSize, blockSize>>>(d_input, d_output, d_lookupTable, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t6 = std::chrono::high_resolution_clock::now();

    cv::Mat result(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(result.data, d_output, imgSize, cudaMemcpyDeviceToHost));
    auto t7 = std::chrono::high_resolution_clock::now();

    cv::imwrite(outputPath, result);

    std::chrono::duration<double, std::milli> h2d = t1 - t0;
    std::chrono::duration<double, std::milli> histKernel = t3 - t2;
    std::chrono::duration<double, std::milli> cdfCompute = t4 - t3;
    std::chrono::duration<double, std::milli> equalizeKernelTime = t6 - t5;
    std::chrono::duration<double, std::milli> d2h = t7 - t6;

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Host->Device transfer:    " << h2d.count() << " ms" << std::endl;
    std::cout << "Histogram kernel (GPU):   " << histKernel.count() << " ms" << std::endl;
    std::cout << "CDF computation (CPU):    " << cdfCompute.count() << " ms" << std::endl;
    std::cout << "Equalize kernel (GPU):    " << equalizeKernelTime.count() << " ms" << std::endl;
    std::cout << "Device->Host transfer:    " << d2h.count() << " ms" << std::endl;
    std::cout << "Total pipeline:           "
               << (h2d + histKernel + cdfCompute + equalizeKernelTime + d2h).count()
               << " ms" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Output saved to: " << outputPath << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_lookupTable);
    cudaFree(d_histogram);

    return 0;
}
