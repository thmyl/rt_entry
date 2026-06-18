import re
import sys

def parse_log(file_path):
    # 存储结果，结构为字典，key是实验组别('1000', '5000', '10000')
    # value 是一个列表，列表中包含该次实验的完整信息
    data = {
        '1000': [],
        '5000': [],
        '10000': []
    }

    # 正则表达式定义
    # 1. 提取实验组别(1000/5000/10000) 和 n_pages
    # 你的日志格式中 centroids_path 在 n_pages 之前
    re_header = re.compile(r"centroids_(\d+)_hilbert.*n_pages\s*=\s*(\d+)")
    
    # 2. 提取 recall100@1024
    re_recall = re.compile(r"recall1000@1024\s*=\s*([\d\.]+)")
    
    # 3. 提取 Used memory
    re_memory = re.compile(r"Used memory:\s*(\d+)")

    # 临时变量用于记录当前块的状态
    current_type = None
    current_pages = None
    current_recall = None

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # --- 步骤1: 识别 Header ---
                if "----------" in line and "centroids_path" in line:
                    match = re_header.search(line)
                    if match:
                        current_type = match.group(1)
                        current_pages = int(match.group(2))
                        # 重置 recall，准备读取新块
                        current_recall = None
                    else:
                        current_type = None # 匹配失败，标记无效
                    continue

                # 如果当前没有在有效的实验块中，跳过
                if current_type is None:
                    continue

                # --- 步骤2: 提取 Recall ---
                if "recall1000@1024" in line:
                    match_r = re_recall.search(line)
                    if match_r:
                        current_recall = float(match_r.group(1))

                # --- 步骤3: 提取 Memory 并保存 ---
                # "Used memory" 通常是块的最后一行信息
                if "Used memory" in line:
                    match_m = re_memory.search(line)
                    if match_m and current_recall is not None:
                        current_memory = int(match_m.group(1))
                        
                        # 直接保存数据到对应组别的列表中
                        if current_type in data:
                            data[current_type].append({
                                'n_pages': current_pages,
                                'memory': current_memory,
                                'recall': current_recall
                            })
                        
                        # 保存后重置状态，防止跨块污染
                        current_type = None
                        current_recall = None

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

    return data

def print_formatted_arrays(data):
    # 定义输出顺序
    experiment_types = ['1000', '5000', '10000']
    
    for exp_type in experiment_types:
        records = data.get(exp_type, [])
        
        if not records:
            continue
            
        # --- 排序 ---
        # 按照 n_pages 降序排列 (15000 -> 14000 -> ... -> 2700)
        # 这样画出来的图点是从右向左或者符合内存从大到小的趋势
        records.sort(key=lambda x: x['n_pages'], reverse=True)
        
        # 提取数组
        mem_list = [r['memory'] for r in records]
        recall_list = [r['recall'] for r in records]
        
        # 打印 Python 数组格式
        print(f"memory{exp_type} = {mem_list}")
        print(f"recall1000_1024_{exp_type} = {recall_list}")
        print("") # 空行分隔

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <log_file>")
        print("Example: python script.py result.log")
    else:
        log_file = sys.argv[1]
        parsed_data = parse_log(log_file)
        if parsed_data:
            print_formatted_arrays(parsed_data)