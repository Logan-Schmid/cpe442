CXX = g++
OPENCV_PKG_CONFIG := $(shell pkg-config --cflags --libs opencv4)
CXXFLAGS = -Werror -Wall -Wpedantic -O0 -std=c++17 -g

TARGET = edge_detector_threads
SRCS = edge_detector_threads.cpp threaded_processing.cpp
INCLS = threaded_processing.hpp
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(OPENCV_PKG_CONFIG)

$(OBJS): $(SRCS) $(INCLS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(OPENCV_PKG_CONFIG)

clean:
	rm -f $(OBJS) $(TARGET)