CXX := g++
CXXFLAGS := -O2 -std=c++17 -Wall

CUDA_HOME := /usr/local/cuda
TRT_INC := /usr/include/aarch64-linux-gnu
TRT_LIB := /usr/lib/aarch64-linux-gnu

INCLUDES := -I$(CUDA_HOME)/include -I$(TRT_INC)
LDFLAGS := -L$(CUDA_HOME)/lib64 -L$(TRT_LIB)
LIBS := -lnvinfer -lcudart

TARGET := app
SRCS := main.cpp yolo_trt.cpp risk_analyzer.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(INCLUDES) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)