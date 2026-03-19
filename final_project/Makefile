# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3 $(shell pkg-config --cflags opencv4)

# Libraries to link: OpenCV + Vulkan/Kompute
LDFLAGS = $(shell pkg-config --libs opencv4) -lkompute -lvulkan

# Project files
TARGET = gpu_edge_detector
SRCS = main.cpp
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

# Rule to compile .comp files into .spv using glslc (with -O for optimization)
%.spv: %.comp
	glslc -O $< -o $@

# Cleanup build artifacts
clean:
	rm -f $(OBJS) $(TARGET) $(SPVS)

.PHONY: all shaders clean