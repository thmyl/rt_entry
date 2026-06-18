import matplotlib.pyplot as plt
import os
import json
import numpy as np
from matplotlib.ticker import FormatStrFormatter  # 记得导入这个库

# 1. 准备数据 (将你提供的 JSON 数据嵌入此处)
json_filename = '100M_recall.json'
with open(json_filename, 'r') as f:
    data = json.load(f)

# 2. 设置绘图风格 (模拟学术论文风格)
# plt.rcParams['font.size'] = 14
# plt.rcParams['axes.titlesize'] = 22
# plt.rcParams['axes.labelsize'] = 20
# plt.rcParams['xtick.labelsize'] = 18
# plt.rcParams['ytick.labelsize'] = 18
# plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 30})  # 基础字体大小
# sz = 45    
sz = 35       
line_w = 3        
marker_sz = 17    

# 定义不同系列的样式：颜色、标记、标签
styles = {
    "1000":  {"color": "#1f77b4", "marker": "^", "label": r"$n_c=1000$"}, # Blue, Triangle Up
    "5000":  {"color": "#ff7f0e", "marker": "v", "label": r"$n_c=5000$"}, # Orange, Triangle Down
    "10000": {"color": "#2ca02c", "marker": "o", "label": r"$n_c=10000$"}, # Green, Circle
    "base":  {"color": "#9467bd", "marker": "s", "label": "Base"}         # Purple, Square
}
# 指定绘制顺序以匹配图例
series_order = ["1000", "5000", "10000", "base"]

# 3. 创建画布和子图
# fig, axes = plt.subplots(2, 2, figsize=(22, 10), constrained_layout=False)
fig, axes = plt.subplots(2, 2, figsize=(22, 16), constrained_layout=False)
# 手动调整布局以为上方图例留出空间
plt.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.98, hspace=0.25, wspace=0.15)

# 数据集和指标的映射
# 格式: (Dataset_Key, Metric_Key, Title, Subplot_Index)
plots_config = [
    ("deep100M", "recall100", "DEEP100M", axes[0, 0]),
    ("sift100M", "recall100", "SIFT100M", axes[0, 1]),
    ("deep100M", "recall1000", None, axes[1, 0]),
    ("sift100M", "recall1000", None, axes[1, 1])
]

# 4. 循环绘图
legend_handles = []
legend_labels = []

for dataset_name, metric_name, title, ax in plots_config:
    dataset = data[dataset_name][metric_name]
    
    for key in series_order:
        series_data = dataset[key]
        
        # 处理数据
        # JSON中的memory单位似乎是MB，参考图中数值为GB (约为40-44)，所以需要除以1024
        x = np.array(series_data["memory"]) / 1024.0
        y = np.array(series_data["recall"])
        
        # 按照 X 轴大小排序，以防连线混乱（虽然原数据似乎大致有序，但排序更保险）
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        if key == "base":
            mask = x_sorted > 39212 / 1024.0
            x_sorted = x_sorted[mask]
            y_sorted = y_sorted[mask]
        
        style = styles[key]
        
        # 绘制线条
        line, = ax.plot(x_sorted, y_sorted, 
                        color=style["color"], 
                        marker=style["marker"], 
                        label=style["label"], 
                        linewidth=line_w, 
                        markersize=marker_sz)
        
        # 仅收集一次图例句柄 (从第一个子图)
        if dataset_name == "deep100M" and metric_name == "recall100":
            legend_handles.append(line)
            legend_labels.append(style["label"])

    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 设置标题 (仅第一行)
    if title:
        ax.set_title(title, fontsize=sz)
        
    # 设置 X 轴标签 (仅第二行)
    if ax in axes[1, :]:
        ax.set_xlabel("Memory (GB)", fontsize=sz)
        
    # 设置 Y 轴标签 (仅第一列)
    # 第一行标签是 Recall@100, 第二行是 Recall@1000
    if ax in axes[:, 0]:
        ylabel = "Recall@100" if metric_name == "recall100" else "Recall@1000"
        ax.set_ylabel(ylabel, fontsize=sz)
    ax.tick_params(axis='both', labelsize=sz)
    # 调整刻度显示
    # SIFT100M 的 X轴范围和 DEEP100M 略有不同，让 matplotlib 自动调整或根据图片微调
    # 这里我们使用自动范围，但确保对齐美观

# 为每个子图单独设置 y 轴刻度
axes[0, 0].set_yticks([0.80, 0.85, 0.90, 0.95])  # 左上
axes[0, 1].set_yticks([0.96, 0.98, 1.00])  # 右上
axes[1, 0].set_yticks([0.40, 0.50, 0.60, 0.70, 0.80])              # 左下
axes[1, 1].set_yticks([0.60, 0.70, 0.80, 0.90, 1.00])              # 右下

axes[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[0, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# 5. 添加顶部共享图例
# loc='lower center' 相对于 bbox_to_anchor 
# bbox_to_anchor=(0.5, 0.88) 将图例放置在整个画布的上方中心
fig.legend(legend_handles, legend_labels, 
           loc='lower center', 
        #    bbox_to_anchor=(0.5, 0.83), # 调整这个Y值以控制图例的垂直位置
           bbox_to_anchor=(0.5, 0.86), # 调整这个Y值以控制图例的垂直位置
           ncol=4, 
           frameon=False, # 无边框
           columnspacing=2.5,
           fontsize=sz) # 增加列间距

# 保存或显示
# plt.savefig("reproduced_chart.png", dpi=300)
output_dir = "./img"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

png_path = os.path.join(output_dir, "2x2.png")
plt.savefig(png_path, format="png", bbox_inches="tight")
print(f"图像已保存到文件：{png_path}")

pdf_path = os.path.join(output_dir, "2x2.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
print(f"图像已保存到文件：{pdf_path}")

plt.show()