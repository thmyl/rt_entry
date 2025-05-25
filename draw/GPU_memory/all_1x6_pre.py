import matplotlib.pyplot as plt
import numpy as np

# 方法标签
methods = ['PCSearch', 'PCSearch+RT', 'CAGRA', 'GGNN', 'GANNS']

# 六个数据集的显存占用（MB）
sift1M = [886, 932, 1074, 1070, 1062]
sift10M = [4216, 4688, 6602, 6580, 6556]
sift100M = [37206, 40814, 0, 0, 0]
deep1M = [884, 930, 950, 946, 938]
deep10M = [5398, 5528, 6564, 6608, 6556]
gist = [1238, 1294, 4358, 4474, 4350]

yticks_list = [
    [700, 800, 900, 1000, 1100],         # sift1M
    [3000, 4000, 5000, 6000],  # sift10M
    [10000, 20000, 30000, 40000],     # sift100M
    [800, 850, 900, 950],         # deep1M
    [4000, 5000, 6000],  # deep10M
    [0, 2000, 4000]   # gist
]

yticklabels_list = [
    # ['700', '800', '900', '1000', '1100'],         # sift1M
    # ['3000', '4000', '5000', '6000'],  # sift10M
    # ['10000', '20000', '30000', '40000'],     # sift100M
    # ['800', '850', '900', '950'],         # deep1M
    # ['4000', '5000', '6000', '7000'],  # deep10M
    # ['0', '2000', '4000', '6000']   # gist
    ['7', '8', '9', '10', '11'],         # sift1M
    ['3', '4', '5', '6'],  # sift10M
    ['1', '2', '3', '4'],     # sift100M
    ['8.0', '8.5', '9.0', '9.5'],         # deep1M
    ['4', '5', '6'],  # deep10M
    ['0', '2', '4']   # gist
]

ylabel_test = [r'$\times$1e2', r'$\times$1e3', r'$\times$1e4', r'$\times$1e2', r'$\times$1e3', r'$\times$1e3']

datasets = [sift1M, sift10M, sift100M, deep1M, deep10M, gist]
dataset_labels = ['sift1M', 'sift10M', 'sift100M', 'deep1M', 'deep10M', 'gist']

# 参数设置
num_methods = len(methods)
num_datasets = len(datasets)
bar_width = 0.6
x = np.arange(num_methods)

# 设置图像
fig, axes = plt.subplots(1, num_datasets, figsize=(18, 4))  # 1行6列
# plt.rcParams.update({'font.size': 25})
plt.rcParams.update({'font.size': 16})  # 所有文字统一为16号字体


# 颜色和纹理
# colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
colors = ['#1F77B4',  # 蓝色
          '#FF7F0E',  # 橙色
          '#2CA02C',  # 绿色
          '#9467bd',  # 紫色
          '#8C564B']  # 棕色
hatch = ['/', '\\', 'x', '.', 'o']  # 纹理

# 把 0 替换为 np.nan，这样就不会画出来，但位置会保留
datasets_cleaned = []
for dataset in datasets:
    cleaned = [value if value != 0 else np.nan for value in dataset]
    datasets_cleaned.append(cleaned)

# 转置数据：按方法为主线画图（method 维度变外层）
for i in range(num_datasets):
    method_values = [datasets_cleaned[i][j] for j in range(num_methods)]
    axes[i].bar(x, method_values,
                width=bar_width,
                color=colors,
                hatch=hatch,
                edgecolor='black',
                linewidth=1.5)
    
    # 设置子图的属性
    axes[i].set_title(dataset_labels[i], fontsize=16)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(methods, rotation=-30, fontsize=12)
    axes[i].set_yticks(yticks_list[i])
    axes[i].set_yticklabels(yticklabels_list[i], fontsize=16)
    axes[i].set_ylim(bottom=yticks_list[i][0])  # 设置y轴下限为第一个刻度
    axes[i].text(-0.1, 1.02, ylabel_test[i], transform=axes[i].transAxes,
                 ha='center', va='bottom', fontsize=16, rotation=0)
    # axes[i].set_yscale("log", base=10)
    if i == 0:  # 只在第一个子图显示纵轴标签
        axes[i].set_ylabel("Memory Usage (MB)", fontsize=16)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.subplots_adjust(wspace=0.1)  # 减少子图水平间距
plt.tight_layout()

# 保存图像
plt.savefig("memory_usage.png", format="png", bbox_inches="tight")
print("图像已保存到文件：memory_usage.png")
