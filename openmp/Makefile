CXX = g++
CXXFLAGS = -O2 -std=c++17

OPENCV_INCLUDE = -I/home/jiitcah.04/.conda/envs/hpcmedvision/include/opencv4
OPENCV_LIB_DIR = /home/jiitcah.04/.conda/envs/hpcmedvision/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

LDFLAGS = -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS) -Wl,-rpath,$(OPENCV_LIB_DIR)

all: preprocess_seq preprocess_omp

preprocess_seq: preprocess.cpp
	$(CXX) $(CXXFLAGS) $(OPENCV_INCLUDE) -o preprocess_seq preprocess.cpp $(LDFLAGS)

preprocess_omp: preprocess.cpp
	$(CXX) $(CXXFLAGS) -fopenmp $(OPENCV_INCLUDE) -o preprocess_omp preprocess.cpp $(LDFLAGS)

clean:
	rm -f preprocess_seq preprocess_omp

.PHONY: all clean
