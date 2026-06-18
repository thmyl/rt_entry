import re
import sys

def parse_log_universal(file_path):
    # 使用字典存储，key 自动生成，value 是数据列表
    data = {}

    # 正则表达式
    # 1. 提取 Header 信息
    # 稍微放宽了正则，尝试提取 centroids_数字，如果提取不到，就用 "default"
    re_header_id = re.compile(r"centroids_(\d+)_hilbert") 
    re_n_pages = re.compile(r"n_pages\s*=\s*(\d+)")
    
    # 2. 提取 recall
    re_recall = re.compile(r"recall100@1024\s*=\s*([\d\.]+)")
    # 3. 提取 memory
    re_memory = re.compile(r"Used memory:\s*(\d+)")

    current_id = "default" # 默认ID
    current_pages = None
    current_recall = None

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # --- 1. 识别 Header ---
                if "----------" in line and "centroids_path" in line:
                    # 尝试从路径中提取 ID (比如 1000, 5000)
                    match_id = re_header_id.search(line)
                    if match_id:
                        current_id = match_id.group(1)
                    else:
                        # 如果路径里没有 centroids_xxx_hilbert 格式，
                        # 如果你只跑一种实验，这里就会统一归为 "default" 组
                        current_id = "experiment" 
                    
                    # 提取 n_pages
                    match_pages = re_n_pages.search(line)
                    if match_pages:
                        current_pages = int(match_pages.group(1))
                        # 重置 recall
                        current_recall = None
                    else:
                        current_pages = None # 这一行虽然像header但没有页数，无效
                    continue

                if current_pages is None:
                    continue

                # --- 2. 提取 Recall ---
                if "recall100@1024" in line:
                    match_r = re_recall.search(line)
                    if match_r:
                        current_recall = float(match_r.group(1))

                # --- 3. 提取 Memory 并保存 ---
                if "Used memory" in line:
                    match_m = re_memory.search(line)
                    if match_m and current_recall is not None:
                        current_memory = int(match_m.group(1))
                        
                        # 动态初始化列表
                        if current_id not in data:
                            data[current_id] = []
                            
                        data[current_id].append({
                            'n_pages': current_pages,
                            'memory': current_memory,
                            'recall': current_recall
                        })
                        
                        # 提交后重置状态
                        current_recall = None

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

    return data

def print_formatted_arrays(data):
    if not data:
        print("No valid data found.")
        return

    # 遍历所有找到的实验组 (无论是 '1000' 还是 'experiment')
    # sorted(data.keys()) 保证输出顺序固定
    for exp_id in sorted(data.keys()):
        records = data[exp_id]
        if not records:
            continue
            
        # 按 n_pages 降序排序
        records.sort(key=lambda x: x['n_pages'], reverse=True)
        
        mem_list = [r['memory'] for r in records]
        recall_list = [r['recall'] for r in records]
        
        print(f"# Experiment Group: {exp_id}")
        print(f"memory_{exp_id} = {mem_list}")
        print(f"recall1000_1024_{exp_id} = {recall_list}")
        print("") 

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <log_file>")
    else:
        log_file = sys.argv[1]
        parsed_data = parse_log_universal(log_file)
        if parsed_data:
            print_formatted_arrays(parsed_data)