import subprocess
import re
import numpy as np

# rt entry

# 编译
# command = "nvcc -D 'OUTFILE=\"out_rt.txt\"' -o query_rt ../query.cu ../RVQ/RVQ.cu ../hybrid/hybrid.cpp ../graph/graph_index/nsw_graph_operations.cu ../functions/distance_kernel.cu ../functions/selectMin1.cu -lcublas -lmkl_rt -DUSE_L2_DIST_"
# result = subprocess.run(command, shell=True, text=True, capture_output=True)
# print(">> stdout:", result.stdout)
# print(">> stderr:", result.stderr)

# 运行
# base_command = ['./../test', '', '']

expand_ratios = [0.8, 0.4, 0.2]
point_ratios = [0.0025, 0.001, 0.0005, 0.0002, 0.0001]
# expand_ratios = [0.8]
# point_ratios = [0.0025]

for i in range(5):
  for expand_ratio in expand_ratios:
    for point_ratio in point_ratios:
      command = base_command.copy()
      command[1] = f'--expand_ratio={expand_ratio}'
      command[2] = f'--point_ratio={point_ratio}'
      print(f"Running command: {' '.join(command)}")
      process = subprocess.Popen(command)
      stdout, stderr = process.communicate()
  with open('out.txt', 'a') as file:
    file.write("\n")
  
# 计算平均值
with open('out.txt', 'r') as file:
  data = file.read()
pca_times = []
rt_times = []
collect_times = []
search_times = []
recall10 = []
for line in data.strip().split('\n'):
    # match = re.match(r'Search time: ([\d.]+) ms\tRecall: ([\d.]+)', line)
    match = re.match(r'time pca projection: ([\d.]+) ms', line)
    if match:
      pca_times.append(float(match.group(1)))

    match = re.match(r'time rt search: ([\d.]+) ms', line)
    if match:
      rt_times.append(float(match.group(1)))

    match = re.match(r'time collect candidates: ([\d.]+) ms', line)
    if match:
      collect_times.append(float(match.group(1)))

    match = re.match(r'time search: ([\d.]+) ms', line)
    if match:
      search_times.append(float(match.group(1)))
    
    match = re.match(r'recall@10 = ([\d.]+) ms', line)
    if match:
      recall10.append(float(match.group(1)))
# search_times = search_times[len(bit_values):]# 去掉第一次运行的值
# recalls = recalls[len(bit_values):]
pca_times = np.array(pca_times).reshape(-1, len(expand_ratios)*len(point_ratios))
rt_times = np.array(rt_times).reshape(-1, len(expand_ratios)*len(point_ratios))
collect_times = np.array(collect_times).reshape(-1, len(expand_ratios)*len(point_ratios))
search_times = np.array(search_times).reshape(-1, len(expand_ratios)*len(point_ratios))
recall10 = np.array(recall10).reshape(-1, len(expand_ratios)*len(point_ratios))

avg_pca_times = np.mean(pca_times, axis=0)
avg_rt_times = np.mean(rt_times, axis=0)
avg_collect_times = np.mean(collect_times, axis=0)
avg_search_times = np.mean(search_times, axis=0)
avg_recall10 = np.mean(recall10, axis=0)

with open('out_avg.txt', 'a') as file:
  file.write("\nAverage PCA Times (ms): {}\n".format(", ".join(f"{x:.4f}" for x in avg_pca_times)))
  file.write("Average RT Times (ms): {}\n".format(", ".join(f"{x:.4f}" for x in avg_rt_times)))
  file.write("Average Collect Times (ms): {}\n".format(", ".join(f"{x:.4f}" for x in avg_collect_times)))
  file.write("Average Search Times (ms): {}\n".format(", ".join(f"{x:.4f}" for x in avg_search_times)))
  file.write("Average Recalls: {}\n".format(", ".join(f"{x:.4f}" for x in avg_recall10)))