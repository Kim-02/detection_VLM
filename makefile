CXX := g++
CXXFLAGS := -O2 -std=c++17 -Wall $(shell pkg-config --cflags opencv4)

CUDA_HOME := /usr/local/cuda
TRT_INC := /usr/include/aarch64-linux-gnu
TRT_LIB := /usr/lib/aarch64-linux-gnu

INCLUDES := -I$(CUDA_HOME)/include -I$(TRT_INC)
LDFLAGS := -L$(CUDA_HOME)/lib64 -L$(TRT_LIB)
LIBS := -lnvinfer -lcudart -lcurl $(shell pkg-config --libs opencv4)

TARGET := app
SRCS := main.cpp yolo_trt.cpp risk_analyzer.cpp image_resize.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)