
NVCC = nvcc
NVCCFLAGS = -O2 -std=c++17

OPENCV_INCLUDE = -I/home/jiitcah.04/.conda/envs/hpcmedvision/include/opencv4
OPENCV_LIB_DIR = /home/jiitcah.04/.conda/envs/hpcmedvision/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

LDFLAGS = -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS) -Xlinker -rpath -Xlinker $(OPENCV_LIB_DIR)

all: gaussian_blur sobel histogram_equalization cuda_benchmark

gaussian_blur: gaussian_blur.cu
	$(NVCC) $(NVCCFLAGS) $(OPENCV_INCLUDE) -o gaussian_blur gaussian_blur.cu $(LDFLAGS)

sobel: sobel.cu
	$(NVCC) $(NVCCFLAGS) $(OPENCV_INCLUDE) -o sobel sobel.cu $(LDFLAGS)

histogram_equalization: histogram_equalization.cu
	$(NVCC) $(NVCCFLAGS) $(OPENCV_INCLUDE) -o histogram_equalization histogram_equalization.cu $(LDFLAGS)

cuda_benchmark: cuda_benchmark.cu
	$(NVCC) $(NVCCFLAGS) $(OPENCV_INCLUDE) -o cuda_benchmark cuda_benchmark.cu $(LDFLAGS)

clean:
	rm -f gaussian_blur sobel histogram_equalization cuda_benchmark

.PHONY: all clean
