CXX = g++
OPENCV_PKG_CONFIG := $(shell pkg-config --cflags --libs opencv4)
CXXFLAGS = -Werror -Wall -Wpedantic -O3 -std=c++17 -g -lpthread

TARGET = edge_detector_vec_instr
SRCS = edge_detector_vec_instr.cpp processing.cpp
INCLS = processing.hpp
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(OPENCV_PKG_CONFIG)

$(OBJS): $(SRCS) $(INCLS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(OPENCV_PKG_CONFIG)

asm:
	$(CXX) $(CXXFLAGS) -S processing.cpp -o processing.s $(OPENCV_PKG_CONFIG)

clean:
	rm -f $(OBJS) $(TARGET) *.s
