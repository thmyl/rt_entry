# Makefile for Page Cache System

NVCC = nvcc
NVCCFLAGS = -std=c++14 -Xcompiler "-Wall -O2 -g"

# 源文件
SRCS = lru.cpp page_cache.cpp
OBJS = $(SRCS:.cpp=.o)

# 测试程序
TEST_SRC = test_cache.cpp
TEST_EXE = test_cache

# 默认目标
all: $(TEST_EXE)

# 编译目标文件
%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 链接测试程序
$(TEST_EXE): $(OBJS) $(TEST_SRC)
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(TEST_SRC) -o $(TEST_EXE)

# 运行测试
test: $(TEST_EXE)
	./$(TEST_EXE)

# 清理
clean:
	rm -f $(OBJS) $(TEST_EXE)

# 重新编译
rebuild: clean all

.PHONY: all test clean rebuild

