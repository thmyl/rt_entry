#!/bin/bash

# 定义要修改的文件名
HEADER_FILE="/home/myl/cache_search/head.h"
TEST_COMMAND="./test --ALGO=1 --topk=0 --n_candidates=1024 --expand_ratio=0.2 --point_ratio=0.00001024 --search_width=4 --t=50 --n_cluster=1000  --max_iter=1000"
TEST_DIR="/home/myl/cache_search/bin/"
BUILD_DIR="/home/myl/cache_search/build/"
# 备份文件
cp "$HEADER_FILE" "${HEADER_FILE}.bak"

echo "已备份 $HEADER_FILE 为 ${HEADER_FILE}.bak"

# 循环从 50 到 30 (步长为 -1)
for i in $(seq 52 -1 20); do
    echo "=================================================="
    echo "正在处理: DIM = $i"
    echo "=================================================="

    # 1. 修改 head.h 文件
    # 使用 sed 正则表达式查找以 '#define DIM ' 开头的行并替换
    # s/^#define DIM .*/.../ 表示替换整行
    sed -i "s/^#define DIM .*/#define DIM $i/" "$HEADER_FILE"

    # 2. 进入 build 文件夹并执行 make
    if [ -d "$BUILD_DIR" ]; then
        cd "$BUILD_DIR"
        # 使用 -j 开启多核编译，加快速度 (根据你的CPU核心数调整，例如 -j8)
        make -j
        
        # 检查 make 是否成功，如果不成功则退出脚本
        if [ $? -ne 0 ]; then
            echo "错误: DIM=$i 时编译失败！脚本已终止。"
            # 尝试恢复文件
            cd ..
            mv "${HEADER_FILE}.bak" "$HEADER_FILE"
            exit 1
        fi
        
        # 返回上一级目录
        cd ..
    else
        echo "错误: 找不到 build 目录"
        exit 1
    fi

    # 3. 进入测试文件所在目录并执行测试
    if [ -d "$TEST_DIR" ]; then
        cd "$TEST_DIR"
        # if [ -f "$TEST_COMMAND" ]; then
            echo "正在运行 $TEST_COMMAND ..."
            $TEST_COMMAND
        # else
        #     echo "错误: 找不到可执行文件 $TEST_COMMAND"
        #     exit 1
        # fi
        # 返回上一级目录
        cd ..
    else
        echo "错误: 找不到测试文件所在目录 $TEST_DIR"
        exit 1
    fi

    echo "完成 DIM = $i 的测试"
    echo ""
    # 可选：如果你想让每次运行间隔几秒，取消下面这行的注释
    # sleep 1
done

# 循环结束后，是否恢复原始文件？(根据需要取消注释)
# mv "${HEADER_FILE}.bak" "$HEADER_FILE"
# echo "已将 $HEADER_FILE 恢复为初始状态"

echo "所有任务执行完毕！"