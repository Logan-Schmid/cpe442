CXX = g++
OPENCV_PKG_CONFIG := $(shell pkg-config --cflags --libs opencv4)
PAPI_FLAGS = -I/usr/local/include -L/usr/local/lib -lpapi
CXXFLAGS = -Werror -Wall -Wpedantic -O1 -std=c++17 -g

TARGET = edge_detector_profiling
SRCS = edge_detector_profiling.cpp processing.cpp
INCLS = processing.hpp
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(OPENCV_PKG_CONFIG) -lpthread $(PAPI_FLAGS)

$(OBJS): $(SRCS) $(INCLS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(OPENCV_PKG_CONFIG) $(PAPI_FLAGS)

asm:
	$(CXX) $(CXXFLAGS) -S processing.cpp -o processing.s $(OPENCV_PKG_CONFIG) $(PAPI_FLAGS)

clean:
	rm -f $(OBJS) $(TARGET) *.s