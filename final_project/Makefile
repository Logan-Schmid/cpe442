# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3 $(shell pkg-config --cflags opencv4)

# Libraries to link: OpenCV, Vulkan, and Kompute
LDFLAGS = $(shell pkg-config --libs opencv4) -lvulkan -lkompute

# Project files
TARGET = edge_detector_final
SRCS = processing.cpp edge_detector_final.cpp
OBJS = $(SRCS:.cpp=.o)

# Shader files
SHADERS = grayscale.comp sobel.comp
SPVS = $(SHADERS:.comp=.spv)

# Default target: Compile shaders first, then the C++ program
all: shaders $(TARGET)

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Compile C++ source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Target to compile all GLSL shaders
shaders: $(SPVS)

# Rule to compile .comp files into .spv using glslc
%.spv: %.comp
	glslc $< -o $@

# Cleanup build artifacts
clean:
	rm -f $(OBJS) $(TARGET) $(SPVS)

.PHONY: all shaders clean