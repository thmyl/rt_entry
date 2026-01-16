import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import os

# 1. 加载数据
json_filename = 'time_breakdown.json'

with open(json_filename, 'r') as f:
    data = json.load(f)

# 2. 数据预处理
datasets = list(data.keys())
pca_times = [data[d]['pca'][0] for d in datasets]
rt_times = [data[d]['rt'][0] for d in datasets]
search_times = [data[d]['search'][0] for d in datasets]

# 3. 设置图表参数
x = np.arange(len(datasets))
width = 0.25 

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(20, 8))
sz = 42

# --- 颜色设置 ---
color_pca = '#4c72b0'
color_rt = '#dd8452'
color_search = '#55a868'

# 4. 绘制柱状图
rects1 = ax.bar(x - width, pca_times, width, label='PCA', 
                color=color_pca, edgecolor='black', hatch='/', linewidth=0.8, alpha=0.9)

rects2 = ax.bar(x, rt_times, width, label='RT Entry Selection', 
                color=color_rt, edgecolor='black', hatch='x', linewidth=0.8, alpha=0.9)

rects3 = ax.bar(x + width, search_times, width, label='Graph Search', 
                color=color_search, edgecolor='black', hatch='\\', linewidth=0.8, alpha=0.9)

# 5. 设置坐标轴和标签 (修改部分)
ax.set_ylabel('Time (ms)', fontsize=sz)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontweight='bold', fontsize=sz-10, rotation=30, ha='center')

# --- 【核心修改】设置对数坐标轴及次要刻度 ---
ax.set_yscale('log')
ax.set_ylim(0.02, 300)

# A. 设置主刻度 (Major Ticks)
ax.set_yticks([0.1, 1, 10, 100])
# 必须显式设置格式化器，否则有时会变成科学计数法
# ax.yaxis.set_major_formatter(ticker.ScalarFormatter()) 

# B. 设置次要刻度 (Minor Ticks)
# base=10.0, subs=... 表示在 0.2, 0.3 ... 0.9 的位置打点
minor_locator = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
ax.yaxis.set_minor_locator(minor_locator)
ax.yaxis.set_minor_formatter(ticker.NullFormatter()) # 次要刻度不显示数字标签

# C. 设置刻度样式 (Tick Params)
# which='major': 设置主刻度样式 (长一点)
# pad: 刻度标签与轴的距离（单位：点），默认约4-6，增大此值可以让标签离轴更远
ax.tick_params(axis='y', which='major', labelsize=sz, length=12, width=1.5, direction='in', pad=10)
# which='minor': 设置次要刻度样式 (短一点，确保它们显示出来)
ax.tick_params(axis='y', which='minor', length=6, width=1, direction='in', color='black')

# 6. 设置图例
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=3, frameon=False, fontsize=sz)

# 7. 网格线
# 仅对主刻度画网格线，如果想次要刻度也画，把 which='major' 改为 which='both'
ax.grid(axis='y', which='major', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('time_breakdown_chart.png', dpi=300, bbox_inches='tight')
plt.show()