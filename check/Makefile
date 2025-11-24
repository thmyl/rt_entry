CUDA_ROOT ?= /usr/local/cuda
CXX := g++
CXXFLAGS := -O3 -std=c++14 -Wall -Wextra -Wpedantic
LDFLAGS :=

INC_DIRS := -I..
SRC_DIR := .
BIN_DIR := bin

TARGET := $(BIN_DIR)/nearest_clusters
SRCS := $(SRC_DIR)/nearest_clusters.cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all clean dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BIN_DIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC_DIRS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC_DIRS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

