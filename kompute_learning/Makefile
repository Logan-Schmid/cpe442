# Compiler and flags
CXX = g++

# Prefer pkg-config metadata for Kompute; fallback to direct libs.
KOMPUTE_CFLAGS = $(shell pkg-config --cflags kompute 2>/dev/null)
KOMPUTE_LIBS = $(shell pkg-config --libs --static kompute 2>/dev/null)
ifeq ($(strip $(KOMPUTE_LIBS)),)
KOMPUTE_LIBS = -lkompute -lvulkan
endif

# Kompute static installs may require explicit fmt linkage.
FMT_LIBS = $(shell pkg-config --libs fmt 2>/dev/null)
ifeq ($(strip $(FMT_LIBS)),)
FMT_LIBS = -lfmt
endif

CXXFLAGS = -std=c++17 -Wall -O1 $(KOMPUTE_CFLAGS)

# Libraries to link: OpenCV + Kompute stack
LDFLAGS = $(KOMPUTE_LIBS) $(FMT_LIBS)

# Project files
TARGET = kompute_demo
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

# # Shader files
# SHADERS = edge_detector.comp
# SPVS = $(SHADERS:.comp=.spv)

# # Default target: Compile shaders first, then the C++ program
# all: shaders $(TARGET)

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Compile C++ source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# # Target to compile all GLSL shaders
# shaders: $(SPVS)

# # Rule to compile .comp files into .spv using glslc
# %.spv: %.comp
# 	glslc -O $< -o $@

# Cleanup build artifacts
clean:
	rm -f $(OBJS) $(TARGET) 
# $(SPVS)

# .PHONY: all shaders clean
