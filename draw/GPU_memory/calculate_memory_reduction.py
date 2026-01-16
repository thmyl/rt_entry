#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 PARS 相比 CAGRA 显存降低的百分比
"""

import os

def calculate_memory_reduction():
    """
    计算每个数据集中 PARS 相比 CAGRA 显存降低的百分比
    """
    # 数据定义（从 all_1x6_small.py 中提取）
    methods = ['PARS-Base', 'PARS', 'CAGRA', 'CAGRA+RT', 'GGNN', 'GANNS']
    datasets_data = [
        [818, 842, 940, 964, 946, 938],       # DEEP1M
        [5332, 5360, 6554, 6590, 6608, 6556], # DEEP10M
        [1174, 1198, 4350, 4376, 4474, 4350], # GIST
        [818, 842, 1064, 1088, 1070, 1062],   # SIFT1M
        [4148, 4238, 6592, 6682, 6580, 6556], # SIFT10M
        [542, 562, 692, 712, 692, 690]        # COCO-I2I
    ]
    dataset_labels = ['DEEP1M', 'DEEP10M', 'GIST', 'SIFT1M', 'SIFT10M', 'COCO-I2I']
    
    # 找到 PARS 和 CAGRA 的索引
    pars_idx = methods.index('PARS')
    cagra_idx = methods.index('CAGRA')
    
    print("=" * 80)
    print(f"{'数据集':<15} {'PARS显存(MB)':<15} {'CAGRA显存(MB)':<15} {'降低百分比(%)':<15}")
    print("=" * 80)
    
    results = {}
    total_reduction = 0
    
    for i, dataset_label in enumerate(dataset_labels):
        pars_memory = datasets_data[i][pars_idx]
        cagra_memory = datasets_data[i][cagra_idx]
        
        # 计算降低百分比
        reduction = ((cagra_memory - pars_memory) / cagra_memory) * 100 if cagra_memory > 0 else 0
        
        results[dataset_label] = {
            'pars_memory': pars_memory,
            'cagra_memory': cagra_memory,
            'reduction_percent': reduction
        }
        
        total_reduction += reduction
        
        # 打印结果
        print(f"{dataset_label:<15} {pars_memory:<15} {cagra_memory:<15} {reduction:<15.2f}")
    
    # 计算平均降低百分比
    avg_reduction = total_reduction / len(dataset_labels)
    
    print("=" * 80)
    print(f"{'平均降低百分比':<15} {'':<15} {'':<15} {avg_reduction:<15.2f}")
    print("=" * 80)
    
    results['average_reduction'] = avg_reduction
    
    return results

if __name__ == "__main__":
    # 计算并显示结果
    results = calculate_memory_reduction()
    
    # 可选：保存结果到 JSON 文件
    import json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "memory_reduction_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")

