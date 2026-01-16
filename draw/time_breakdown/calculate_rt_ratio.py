#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算每个数据集中 rt 时间占总时间的比例
"""

import json
import os

def calculate_rt_ratio(json_file):
    """
    计算每个数据集中 rt 时间占总时间的比例
    
    Args:
        json_file: JSON 文件路径
    """
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 60)
    print(f"{'数据集':<15} {'RT时间':<12} {'总时间':<12} {'RT占比(%)':<12}")
    print("=" * 60)
    
    results = {}
    
    for dataset, times in data.items():
        # 提取各个时间值（从列表中取第一个元素）
        pca_time = times['pca'][0]
        rt_time = times['rt'][0]
        search_time = times['search'][0]
        
        # 计算总时间
        total_time = pca_time + rt_time + search_time
        
        # 计算 rt 占比
        rt_ratio = (rt_time / total_time) * 100 if total_time > 0 else 0
        
        results[dataset] = {
            'rt_time': rt_time,
            'total_time': total_time,
            'rt_ratio': rt_ratio
        }
        
        # 打印结果
        print(f"{dataset:<15} {rt_time:<12.6f} {total_time:<12.6f} {rt_ratio:<12.2f}")
    
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, "time_breakdown.json")
    
    # 计算并显示结果
    results = calculate_rt_ratio(json_file)
    
    # 可选：保存结果到 JSON 文件
    output_file = os.path.join(script_dir, "rt_ratio_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")

