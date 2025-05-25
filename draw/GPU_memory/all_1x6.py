import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
plt.rcParams['font.family'] = 'Times New Roman'

# 方法标签
methods = ['PCSearch', 'PCSearch+RT', 'CAGRA', 'CAGRA+RT', 'GGNN', 'GANNS']

# 六个数据集的显存占用（MB）

# sift100M = [37140, 37676, 0, 0, 0, 0] #
deep1M = [818, 842, 940, 964, 946, 938] #
deep10M = [5332, 5360, 6554, 6590, 6608, 6556] #
gist = [1174, 1198, 4350, 4376, 4474, 4350] #
sift1M = [818, 842, 1064, 1088, 1070, 1062] #
sift10M = [4148, 4238, 6592, 6682, 6580, 6556] #
COCO_I2I = [542, 562, 692, 712, 692, 690] #

yticks_list = [
    [700, 800, 900, 1000],         # deep1M
    [4000, 5000, 6000, 7000],  # deep10M
    [0, 2000, 4000],   # gist
    [700, 800, 900, 1000, 1100],         # sift1M
    [3000, 4000, 5000, 6000, 7000],  # sift10M
    # [10000, 20000, 30000, 40000],     # sift100M
    [500, 600, 700], # COCO_I2I
    
]

yticklabels_list = [
    # ['700', '800', '900', '1000', '1100'],         # sift1M
    # ['3000', '4000', '5000', '6000'],  # sift10M
    # ['10000', '20000', '30000', '40000'],     # sift100M
    # ['800', '850', '900', '950'],         # deep1M
    # ['4000', '5000', '6000', '7000'],  # deep10M
    # ['0', '2000', '4000', '6000']   # gist

    ['7.0', '8.0', '9.0', '10'],         # deep1M
    ['4', '5', '6', '7'],  # deep10M
    ['0', '2', '4'],   # gist
    ['7', '8', '9', '10', '11'],         # sift1M
    ['3', '4', '5', '6', '7'],  # sift10M
    # ['1', '2', '3', '4'],     # sift100M
    ['5', '6', '7'], # COCO_I2I
]

ylabel_test = [r'$\times$1e2', r'$\times$1e3', r'$\times$1e4', r'$\times$1e2', r'$\times$1e3', r'$\times$1e2']

datasets = [deep1M, deep10M, gist, sift1M, sift10M, COCO_I2I]
dataset_labels = ['DEEP-1M', 'DEEP-10M', 'GIST', 'SIFT-1M', 'SIFT-10M', 'COCO-I2I']

# 参数设置
num_methods = len(methods)
num_datasets = len(datasets)
bar_width = 0.6
x = np.arange(num_methods)

# 设置图像
fig, axes = plt.subplots(1, num_datasets, figsize=(18, 4))  # 1行6列
# plt.rcParams.update({'font.size': 25})
sz=25
plt.rcParams.update({'font.size': 25})  # 所有文字统一为16号字体


# 颜色和纹理
# colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
colors = ['#1F77B4',  # 蓝色
          '#FF7F0E',  # 橙色
          '#2CA02C',  # 绿色
          '#D62728',  # 红色
          '#9467bd',  # 紫色
          '#8C564B']  # 棕色
hatch = ['/', '\\', 'x', '.', '++', 'o']  # 纹理

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
    axes[i].set_title(dataset_labels[i], fontsize=sz)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(range(1, 7), rotation=0, ha='center', fontsize=sz)
    # axes[i].set_xticklabels(methods, rotation=-60, ha='left', fontsize=12)
    axes[i].set_yticks(yticks_list[i])
    axes[i].set_yticklabels(yticklabels_list[i], fontsize=sz)
    axes[i].set_ylim(bottom=yticks_list[i][0])  # 设置y轴下限为第一个刻度
    axes[i].text(0, 1.07, ylabel_test[i], transform=axes[i].transAxes,
                 ha='center', va='bottom', fontsize=sz-5, rotation=0)
    # axes[i].set_yscale("log", base=10)
    if i == 0:  # 只在第一个子图显示纵轴标签
        axes[i].set_ylabel("Memory Usage (MB)", fontsize=sz)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    axes[i].set_axisbelow(True)

# 调整布局
plt.subplots_adjust(wspace=0.1)  # 减少子图水平间距
# plt.tight_layout()
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 留出顶部空间
plt.subplots_adjust(left=0.05, right=0.98, top=0.72, bottom=0.1, wspace=0.2)

legend_handles = [mpatches.Patch(facecolor=colors[i], edgecolor='black',
                                  hatch=hatch[i], label=methods[i]) for i in range(num_methods)]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=6, fontsize=sz, frameon=False)
# 保存图像
plt.savefig("memory_usage.png", format="png", bbox_inches="tight")
print("图像已保存到文件：memory_usage.png")

plt.savefig("memory_usage.pdf", format="pdf", bbox_inches="tight")
print("图像已保存到文件：memory_usage.pdf")
