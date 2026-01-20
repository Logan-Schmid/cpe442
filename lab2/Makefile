CXX = g++
OPENCV_PKG_CONFIG := $(shell pkg-config --cflags --libs opencv4)
CXXFLAGS = -Werror -Wall -Wpedantic -O0 -std=c++17

TARGET = image_viewer
SRCS = image_viewer.cpp
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) $(SRCS) $(CXXFLAGS) -o $(TARGET) $(OPENCV_PKG_CONFIG)

$(OBJS): $(SRCS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(OPENCV_PKG_CONFIG)

clean:
	rm -f $(OBJS) $(TARGET)