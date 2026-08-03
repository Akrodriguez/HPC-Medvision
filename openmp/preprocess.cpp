/*
 * preprocess.cpp
 * HPC-MedVision - Phase 3/4: Sequential and OpenMP MRI Preprocessing
 *
 * Performs on each MRI slice image (.tif, excluding masks):
 *   1. Load image from disk
 *   2. Resize to fixed dimensions (256x256)
 *   3. Normalize pixel values to [0.0, 1.0]
 *   4. Compute grayscale intensity histogram (256 bins)
 *   5. Save resized image to output directory
 *
 * Compile WITHOUT -fopenmp  -> runs sequentially (Phase 3 baseline)
 * Compile WITH    -fopenmp  -> runs in parallel   (Phase 4)
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

const std::string RAW_DIR = "../data/raw/kaggle_3m";
const std::string PROCESSED_DIR = "../data/processed";
const int TARGET_WIDTH = 256;
const int TARGET_HEIGHT = 256;

std::vector<std::string> collectImagePaths(const std::string& rootDir) {
    std::vector<std::string> paths;
    for (const auto& entry : fs::recursive_directory_iterator(rootDir)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();
        std::string filename = entry.path().filename().string();

        if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".tif") {
            if (filename.find("_mask") == std::string::npos) {
                paths.push_back(path);
            }
        }
    }
    return paths;
}

bool processImage(const std::string& inputPath, const std::string& rawRoot,
                   const std::string& outRoot) {
    cv::Mat img = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "WARNING: Failed to load " << inputPath << std::endl;
        return false;
    }

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, cv::INTER_LINEAR);

    cv::Mat normalized;
    resized.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

    int histogram[256] = {0};
    for (int r = 0; r < gray.rows; r++) {
        for (int c = 0; c < gray.cols; c++) {
            unsigned char pixel = gray.at<unsigned char>(r, c);
            histogram[pixel]++;
        }
    }

    fs::path relative = fs::relative(inputPath, rawRoot);
    fs::path outPath = fs::path(outRoot) / relative;
    fs::create_directories(outPath.parent_path());

    if (!cv::imwrite(outPath.string(), resized)) {
        std::cerr << "WARNING: Failed to write " << outPath.string() << std::endl;
        return false;
    }

    return true;
}

int main() {
    std::cout << "=== HPC-MedVision Preprocessing ===" << std::endl;

#ifdef _OPENMP
    std::cout << "Mode: OpenMP PARALLEL" << std::endl;
    std::cout << "Max threads available: " << omp_get_max_threads() << std::endl;
#else
    std::cout << "Mode: SEQUENTIAL" << std::endl;
#endif

    std::cout << "Scanning: " << RAW_DIR << std::endl;
    std::vector<std::string> imagePaths = collectImagePaths(RAW_DIR);
    int n = static_cast<int>(imagePaths.size());
    std::cout << "Found " << n << " MRI slice images to process." << std::endl;

    if (n == 0) {
        std::cerr << "ERROR: No images found. Check RAW_DIR path." << std::endl;
        return 1;
    }

    fs::create_directories(PROCESSED_DIR);

    int successCount = 0;
    int failCount = 0;

    auto startTime = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic) reduction(+:successCount, failCount)
    for (int i = 0; i < n; i++) {
        bool ok = processImage(imagePaths[i], RAW_DIR, PROCESSED_DIR);
        if (ok) {
            successCount++;
        } else {
            failCount++;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Images processed successfully: " << successCount << std::endl;
    std::cout << "Images failed: " << failCount << std::endl;
    std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Average time per image: " << (elapsed.count() / n) << " seconds" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    return 0;
}

