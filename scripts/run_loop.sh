#!/bin/bash

OUTPUT_FILE="all_output.txt"

# 清空旧结果
> "$OUTPUT_FILE"

# 从 1 到 100 循环
for i in $(seq 1 100); do
    echo "Running iteration $i..." | tee -a "$OUTPUT_FILE"
    bash run_all.sh 100 10000 $i 1 64 500 100 100 >> "$OUTPUT_FILE" 2>&1
    echo "Done iteration $i" | tee -a "$OUTPUT_FILE"
    echo "----------------------------------------" >> "$OUTPUT_FILE"
done
