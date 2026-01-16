import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import os

# 1. 加载数据 (模拟部分，请确保目录下有 json 文件)
json_filename = 'time_breakdown.json'

with open(json_filename, 'r') as f:
    data = json.load(f)

# 2. 数据预处理
datasets = list(data.keys())
pca_times = np.array([data[d]['pca'][0] for d in datasets])
rt_times = np.array([data[d]['rt'][0] for d in datasets])
search_times = np.array([data[d]['search'][0] for d in datasets])

# 计算总时间和百分比
total_times = pca_times + rt_times + search_times
pca_pct = (pca_times / total_times) * 100
rt_pct = (rt_times / total_times) * 100
search_pct = (search_times / total_times) * 100

# 3. 设置图表布局
plt.rcParams['font.family'] = 'Times New Roman'
sz = 36 # 稍微调小一点点字体以适应两张图，或者保持42看效果

# 创建两个子图：ax1(上部分-堆叠图), ax2(下部分-原来的图)
# height_ratios=[1, 1.2] 让下面那个稍微高一点点（因为有复杂的x轴标签）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 7), sharex=True, 
                               gridspec_kw={'height_ratios': [1, 1.2], 'hspace': 0.05})

x = np.arange(len(datasets))

# --- 颜色设置 ---
color_pca = '#4c72b0'
color_rt = '#dd8452'
color_search = '#55a868'

# ==========================================
# 上部分：堆叠柱状图 (Percentage Breakdown)
# ==========================================
width_stack = 0.6 # 堆叠图稍微宽一点，好看

# 绘制堆叠柱
# 底部是 PCA
p1 = ax1.bar(x, pca_pct, width_stack, label='PCA', 
             color=color_pca, edgecolor='black', hatch='/', alpha=0.9)
# 中间是 RT (bottom=pca_pct)
p2 = ax1.bar(x, rt_pct, width_stack, bottom=pca_pct, label='RT Entry Selection', 
             color=color_rt, edgecolor='black', hatch='x', alpha=0.9)
# 顶部是 Search (bottom=pca_pct + rt_pct)
p3 = ax1.bar(x, search_pct, width_stack, bottom=pca_pct + rt_pct, label='Graph Search', 
             color=color_search, edgecolor='black', hatch='\\', alpha=0.9)

# 添加百分比文字标注
# 阈值：如果百分比太小(比如小于3%)，就不显示文字，避免挤在一起
text_threshold = 3.0 

for i in range(len(datasets)):
    # 1. PCA 文字
    if pca_pct[i] > text_threshold:
        ax1.text(x[i], pca_pct[i]/2, f"{pca_pct[i]:.0f}%", 
                 ha='center', va='center', fontsize=sz-8, color='white', fontweight='bold')
    
    # 2. RT 文字 (位置在 PCA高度 + RT高度的一半)
    if rt_pct[i] > text_threshold:
        ax1.text(x[i], pca_pct[i] + rt_pct[i]/2, f"{rt_pct[i]:.0f}%", 
                 ha='center', va='center', fontsize=sz-8, color='white', fontweight='bold')
    
    # 3. Search 文字 (位置在 PCA+RT高度 + Search高度的一半)
    if search_pct[i] > text_threshold:
        ax1.text(x[i], pca_pct[i] + rt_pct[i] + search_pct[i]/2, f"{search_pct[i]:.0f}%", 
                 ha='center', va='center', fontsize=sz-8, color='white', fontweight='bold')

# 上图设置
ax1.set_ylabel('Ratio (%)', fontsize=sz)
ax1.set_ylim(0, 100)
ax1.tick_params(axis='y', labelsize=sz-5, length=10, width=1.5, direction='in', pad=10)
# 隐藏上图的X轴刻度线（只留标签在下图）
ax1.tick_params(axis='x', length=0) 

# ==========================================
# 下部分：分组柱状图 (Absolute Time Log Scale)
# ==========================================
width = 0.25 

rects1 = ax2.bar(x - width, pca_times, width, label='PCA', 
                color=color_pca, edgecolor='black', hatch='/', linewidth=0.8, alpha=0.9)
rects2 = ax2.bar(x, rt_times, width, label='RT Entry Selection', 
                color=color_rt, edgecolor='black', hatch='x', linewidth=0.8, alpha=0.9)
rects3 = ax2.bar(x + width, search_times, width, label='Graph Search', 
                color=color_search, edgecolor='black', hatch='\\', linewidth=0.8, alpha=0.9)

# 下图设置
ax2.set_ylabel('Time (ms)', fontsize=sz)
ax2.set_xticks(x)
# 这里的 rotation=30 解决标签挤在一起的问题
ax2.set_xticklabels(datasets, fontweight='bold', fontsize=sz-8, rotation=30, ha='center')

# 对数坐标轴设置
ax2.set_yscale('log')
ax2.set_ylim(0.02, 300)
ax2.set_yticks([0.1, 1, 10, 100])
minor_locator = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
ax2.yaxis.set_minor_locator(minor_locator)
ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
ax2.tick_params(axis='y', which='major', labelsize=sz-5, length=12, width=1.5, direction='in', pad=10)
ax2.tick_params(axis='y', which='minor', length=6, width=1, direction='in')

# ==========================================
# 全局设置：图例与保存
# ==========================================

# 只需要一个图例，放在最上面
# handles和labels取自上图即可（因为颜色是一样的）
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, frameon=False, fontsize=sz)

# 网格线
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax2.grid(axis='y', which='major', linestyle='--', alpha=0.5)

plt.tight_layout()
# 因为加了 bbox_to_anchor 的图例，tight_layout 可能会切掉图例，留出顶部空间
plt.subplots_adjust(top=0.90) 

plt.savefig('time_breakdown_combined.png', dpi=300, bbox_inches='tight')
plt.savefig('time_breakdown_combined.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()