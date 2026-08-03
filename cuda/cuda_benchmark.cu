
/*
 * cuda_benchmark.cu
 * Phase 8: CUDA kernel benchmarking with cold-start separation.
 * Processes N images in one CUDA context. Image 0 = "cold" (includes
 * init cost). Remaining images = "warm" (true steady-state kernel cost).
 *
 * Usage: ./cuda_benchmark <images_dir> <num_images> <output_csv>
 */

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__ \
                   << " -> " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

__constant__ float d_gaussianKernel[25] = {
    1,4,7,4,1, 4,16,26,16,4, 7,26,41,26,7, 4,16,26,16,4, 1,4,7,4,1
};
const float GAUSSIAN_SUM = 273.0f;

__constant__ int d_sobelGx[9] = {-1,0,1,-2,0,2,-1,0,1};
__constant__ int d_sobelGy[9] = {-1,-2,-1,0,0,0,1,2,1};

#define HIST_SIZE 256

__global__ void gaussianBlurKernel(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=w||y>=h) return;
    float sum=0.0f;
    for (int ky=-2; ky<=2; ky++)
        for (int kx=-2; kx<=2; kx++) {
            int px=min(max(x+kx,0),w-1), py=min(max(y+ky,0),h-1);
            sum += d_gaussianKernel[(ky+2)*5+(kx+2)] * in[py*w+px];
        }
    out[y*w+x] = (unsigned char)(sum/GAUSSIAN_SUM);
}

__global__ void sobelKernel(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=w||y>=h) return;
    float gx=0,gy=0;
    for (int ky=-1; ky<=1; ky++)
        for (int kx=-1; kx<=1; kx++) {
            int px=min(max(x+kx,0),w-1), py=min(max(y+ky,0),h-1);
            float v = in[py*w+px];
            int idx=(ky+1)*3+(kx+1);
            gx += d_sobelGx[idx]*v;
            gy += d_sobelGy[idx]*v;
        }
    float mag = sqrtf(gx*gx+gy*gy);
    mag = fminf(fmaxf(mag,0.0f),255.0f);
    out[y*w+x] = (unsigned char)mag;
}

__global__ void histogramKernel(const unsigned char* in, int* hist, int w, int h) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=w||y>=h) return;
    atomicAdd(&hist[in[y*w+x]], 1);
}

__global__ void equalizeKernel(const unsigned char* in, unsigned char* out, const unsigned char* lut, int w, int h) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=w||y>=h) return;
    out[y*w+x] = lut[in[y*w+x]];
}

double runBlur(unsigned char* d_in, unsigned char* d_out, int w, int h) {
    dim3 block(16,16), grid((w+15)/16,(h+15)/16);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    gaussianBlurKernel<<<grid,block>>>(d_in, d_out, w, h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1-t0).count();
}

double runSobel(unsigned char* d_in, unsigned char* d_out, int w, int h) {
    dim3 block(16,16), grid((w+15)/16,(h+15)/16);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    sobelKernel<<<grid,block>>>(d_in, d_out, w, h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1-t0).count();
}

double runHistEq(unsigned char* d_in, unsigned char* d_out, int* d_hist, unsigned char* d_lut, int w, int h, int totalPixels) {
    dim3 block(16,16), grid((w+15)/16,(h+15)/16);
    CUDA_CHECK(cudaMemset(d_hist, 0, HIST_SIZE*sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    histogramKernel<<<grid,block>>>(d_in, d_hist, w, h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int histogram[HIST_SIZE];
    CUDA_CHECK(cudaMemcpy(histogram, d_hist, HIST_SIZE*sizeof(int), cudaMemcpyDeviceToHost));
    long cdf[HIST_SIZE];
    cdf[0]=histogram[0];
    for(int i=1;i<HIST_SIZE;i++) cdf[i]=cdf[i-1]+histogram[i];
    long cdfMin=0;
    for(int i=0;i<HIST_SIZE;i++) if(cdf[i]!=0){cdfMin=cdf[i];break;}
    unsigned char lut[HIST_SIZE];
    for(int i=0;i<HIST_SIZE;i++){
        if(totalPixels-cdfMin==0) lut[i]=0;
        else {
            float n=(float)(cdf[i]-cdfMin)/(float)(totalPixels-cdfMin)*255.0f;
            n=fminf(fmaxf(n,0.0f),255.0f);
            lut[i]=(unsigned char)n;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_lut, lut, HIST_SIZE*sizeof(unsigned char), cudaMemcpyHostToDevice));
    equalizeKernel<<<grid,block>>>(d_in, d_out, d_lut, w, h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1-t0).count();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <images_dir> <num_images> <output_csv>" << std::endl;
        return 1;
    }
    std::string imagesDir = argv[1];
    int numImages = std::stoi(argv[2]);
    std::string outputCsv = argv[3];

    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::recursive_directory_iterator(imagesDir)) {
        if (!entry.is_regular_file()) continue;
        std::string fn = entry.path().filename().string();
        if (fn.size()>=4 && fn.substr(fn.size()-4)==".tif" && fn.find("_mask")==std::string::npos) {
            imagePaths.push_back(entry.path().string());
            if ((int)imagePaths.size() >= numImages) break;
        }
    }
    std::cout << "Collected " << imagePaths.size() << " sample images." << std::endl;

    std::ofstream csv(outputCsv);
    csv << "kernel,image_index,phase,kernel_time_ms\n";

    int W=256, H=256;
    size_t imgSize = W*H*sizeof(unsigned char);

    unsigned char *d_in, *d_out, *d_lut;
    int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_in, imgSize));
    CUDA_CHECK(cudaMalloc(&d_out, imgSize));
    CUDA_CHECK(cudaMalloc(&d_lut, HIST_SIZE*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_hist, HIST_SIZE*sizeof(int)));

    for (size_t i=0; i<imagePaths.size(); i++) {
        cv::Mat img = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        cv::resize(img, img, cv::Size(W, H));
        CUDA_CHECK(cudaMemcpy(d_in, img.data, imgSize, cudaMemcpyHostToDevice));

        std::string phase = (i==0) ? "cold" : "warm";

        double blurT = runBlur(d_in, d_out, W, H);
        csv << "gaussian_blur," << i << "," << phase << "," << blurT << "\n";

        double sobelT = runSobel(d_in, d_out, W, H);
        csv << "sobel," << i << "," << phase << "," << sobelT << "\n";

        double histT = runHistEq(d_in, d_out, d_hist, d_lut, W, H, W*H);
        csv << "histogram_equalization," << i << "," << phase << "," << histT << "\n";

        std::cout << "Image " << i << " (" << phase << "): blur=" << blurT
                   << "ms sobel=" << sobelT << "ms histeq=" << histT << "ms" << std::endl;
    }

    csv.close();
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_hist);
    std::cout << "Results saved to " << outputCsv << std::endl;
    return 0;
}
