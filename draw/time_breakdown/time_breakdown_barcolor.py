import matplotlib.pyplot as plt
import numpy as np
import json
import os

# 1. 加载数据
json_filename = 'time_breakdown.json'

with open(json_filename, 'r') as f:
    data = json.load(f)
    
# 2. 数据准备
datasets = list(data.keys())
pca_times = [data[d]['pca'][0] for d in datasets]
rt_times = [data[d]['rt'][0] for d in datasets]
search_times = [data[d]['search'][0] for d in datasets]

# 3. 设置图表参数
x = np.arange(len(datasets))
width = 0.25 

plt.rcParams['font.family'] = 'sans-serif' # 彩色图通常用无衬线字体看着更现代
# 如果你需要保持论文风格的衬线字体，可以把上面这行改成 'serif'

fig, ax = plt.subplots(figsize=(12, 6))

# --- 颜色设置 ---
# 这里选择了三种对比度高的柔和颜色
color_pca = '#4c72b0'    # 蓝色
color_rt = '#dd8452'     # 橙色
color_search = '#55a868' # 绿色

# 4. 绘制柱状图 (移除 hatch，使用 color)
# alpha=0.9 让颜色稍微透一点点，看着不刺眼
rects1 = ax.bar(x - width, pca_times, width, label='PCA', 
                color=color_pca, edgecolor='black', linewidth=0.8, alpha=0.9)

rects2 = ax.bar(x, rt_times, width, label='RT Entry Selection', 
                color=color_rt, edgecolor='black', linewidth=0.8, alpha=0.9)

rects3 = ax.bar(x + width, search_times, width, label='Graph Search', 
                color=color_search, edgecolor='black', linewidth=0.8, alpha=0.9)

# 5. 坐标轴设置
ax.set_ylabel('Time Consumption (Sec)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11, rotation=0) # 如果标签太长可以改 rotation=30

# 对数坐标轴
ax.set_yscale('log')
ax.set_ylim(0.02, 300) # 根据数据范围调整

# 6. 图例设置
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, frameon=False, fontsize=12)

# 7. 网格线与外观优化
ax.tick_params(direction='in', which='both', length=4)
# 添加淡灰色的水平网格线，辅助阅读对数坐标
ax.grid(axis='y', which='major', linestyle='--', alpha=0.4, color='gray')
ax.set_axisbelow(True) # 让网格线在柱子后面

plt.tight_layout()

# 保存并显示
plt.savefig('time_breakdown_color.png', dpi=300, bbox_inches='tight')
plt.show()