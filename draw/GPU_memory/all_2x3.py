import matplotlib.pyplot as plt
import numpy as np

# 方法标签
methods = ['PCSearch', 'RT-PCSearch', 'CAGRA', 'GGNN', 'GANNS']

# 六个数据集的显存占用（MB）
sift1M = [886, 932, 1074, 1070, 0]
sift10M = [4216, 4688, 6602, 6580, 0]
sift100M = [37206, 40814, 0, 0, 0]
deep1M = [884, 930, 950, 946, 0]
deep10M = [5398, 5528, 6564, 6608, 0]
gist = [1238, 1294, 4358, 4474, 0]

datasets = [sift1M, sift10M, sift100M, deep1M, deep10M, gist]
dataset_labels = ['sift1M', 'sift10M', 'sift100M', 'deep1M', 'deep10M', 'gist']

# 参数设置
num_methods = len(methods)
num_datasets = len(datasets)
bar_width = 0.13
x = np.arange(num_methods)

# 设置图像
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2行3列

# 颜色和纹理
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
hatch = ['/', '\\', 'x', 'o', '.']

# 把 0 替换为 np.nan，这样就不会画出来，但位置会保留
datasets_cleaned = []
for dataset in datasets:
    cleaned = [value if value != 0 else np.nan for value in dataset]
    datasets_cleaned.append(cleaned)

# 转置数据：按方法为主线画图（method 维度变外层）
for i in range(num_datasets):
    method_values = [datasets_cleaned[i][j] for j in range(num_methods)]
    row, col = divmod(i, 3)  # 获取子图位置
    axes[row, col].bar(x, method_values,
                       width=bar_width,
                       color=colors,
                       hatch=hatch,
                       edgecolor='black',
                       linewidth=1.5)
    
    # 设置子图的属性
    axes[row, col].set_title(dataset_labels[i])
    axes[row, col].set_xticks(x)
    axes[row, col].set_xticklabels(methods, rotation=-25)
    axes[row, col].set_ylabel("Memory Usage (MB)")
    axes[row, col].grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig("grouped_bar_chart_separate_plots.png", format="png", bbox_inches="tight")
print("图像已保存到文件：grouped_bar_chart_separate_plots.png")
