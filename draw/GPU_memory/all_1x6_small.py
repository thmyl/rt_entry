import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# 1. 全局风格设置 (针对缩小显示优化)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 2.5   # 极粗的边框，缩小时才看得清
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 2.5 # 刻度线加粗
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams.update({'font.size': 36}) # 巨型字号

# 2. 数据准备
methods = ['PARS-Base', 'PARS', 'CAGRA', 'CAGRA+RT', 'GGNN', 'GANNS']
datasets_data = [
    [818, 842, 940, 964, 946, 938],       # DEEP1M
    [5332, 5360, 6554, 6590, 6608, 6556], # DEEP10M
    [1174, 1198, 4350, 4376, 4474, 4350], # GIST
    [818, 842, 1064, 1088, 1070, 1062],   # SIFT1M
    [4148, 4238, 6592, 6682, 6580, 6556], # SIFT10M
    [542, 562, 692, 712, 692, 690]        # COCO-I2I
]
dataset_labels = ['DEEP1M', 'DEEP10M', 'GIST', 'SIFT1M', 'SIFT10M', 'COCO-I2I']

# 刻度设置
yticks_list = [
    [700, 800, 900, 1000],          
    [4000, 5000, 6000, 7000],       
    [0, 2000, 4000],                
    [700, 800, 900, 1000, 1100],    
    [3000, 4000, 5000, 6000, 7000], 
    [500, 600, 700]                 
]

yticklabels_list = [
    ['7', '8', '9', '10'],          
    ['4', '5', '6', '7'],           
    ['0', '2', '4'],                
    ['7', '8', '9', '10', '11'],    
    ['3', '4', '5', '6', '7'],      
    ['5', '6', '7']                 
]

scale_labels = [r'$\times 10^2$', r'$\times 10^3$', r'$\times 10^3$', 
                r'$\times 10^2$', r'$\times 10^3$', r'$\times 10^2$']

# 3. 绘图参数
num_datasets = len(datasets_data)
num_methods = len(methods)
x = np.arange(num_methods)
bar_width = 0.75 # 稍微宽一点，填满瘦长的空间
# hatchs = ['//', '\\\\', 'xx', '..', '++', 'oo']
colors = ['#1F77B4',  # 蓝色
          '#FF7F0E',  # 橙色
          '#2CA02C',  # 绿色
          '#D62728',  # 红色
          '#9467bd',  # 紫色
          '#8C564B']  # 棕色
hatchs = ['/', '\\', 'x', '.', '+', 'o']  # 纹理 - 增加密度使其更明显

# 4. 创建画布：关键点在这里
# figsize=(25, 12) -> 宽度25，高度12
# 意味着每个子图大约宽 4 inch，高 12 inch -> 1:3 的瘦高比例
fig, axes = plt.subplots(1, num_datasets, figsize=(25, 6)) 

# 5. 循环绘制
for i, ax in enumerate(axes):
    data = datasets_data[i]
    
    # 绘制 - 为每个方法使用不同的颜色和纹理
    bars = ax.bar(x, data,
                  width=bar_width,
                  color=colors,             # 每个方法使用不同颜色
                  edgecolor='black',   
                  linewidth=2.0)            # 柱子边框加粗
    
    # 为每个柱添加不同的纹理
    for bar, pattern in zip(bars, hatchs):
        bar.set_hatch(pattern)

    # 标题
    # ax.set_title(dataset_labels[i], fontsize=45, fontweight='bold', pad=20)
    ax.set_xlabel(dataset_labels[i], fontsize=45, fontweight='bold', labelpad=20)
    
    # Y轴设置
    ax.set_yticks(yticks_list[i])
    ax.set_yticklabels(yticklabels_list[i], fontsize=50) # 刻度字体超大
    
    y_min = yticks_list[i][0]
    y_max = yticks_list[i][-1]
    y_range = y_max - y_min
    ax.set_ylim(bottom=y_min, top=y_max + y_range * 0.15) 

    # 角标 (放在左上角内部，字号极大)
    ax.text(-0.05, 1.15, scale_labels[i], transform=ax.transAxes,
            ha='left', va='top', fontsize=50, rotation=0,
            bbox=dict(facecolor='none', edgecolor='none', alpha=0.6))

    # 网格线
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray', linewidth=2.0)
    ax.set_axisbelow(True)

    # 隐藏X轴刻度
    ax.set_xticks([]) 
    
    # 加粗边框 (Spines)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        
    # 只在第一个图显示Y轴标题
    if i == 0:
        ax.set_ylabel("Memory (MB)", fontsize=50, labelpad=20)

# 6. 调整布局
# wspace=0.35 增加左右间距，防止大号字体重叠
plt.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.05, wspace=0.35)

# 7. 全局图例 (1行6列，放在最顶上)
legend_handles = [mpatches.Patch(facecolor=colors[i], edgecolor='black', linewidth=2.0,
                                  hatch=hatchs[i], label=methods[i]) for i in range(num_methods)]

fig.legend(handles=legend_handles, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 1.1), 
           ncol=6,                     # 一行排开
           fontsize=40,                # 图例字体极大
           frameon=False,
           columnspacing=1.2,
           handlelength=1.5, 
           handleheight=0.7)

# 8. 保存
plt.savefig("memory_usage_tall_1x6.png", format="png", bbox_inches="tight", dpi=300)
print("图像已保存到文件：memory_usage_tall_1x6.png")

plt.savefig("memory_usage_tall_1x6.pdf", format="pdf", bbox_inches="tight")
print("图像已保存到文件：memory_usage_tall_1x6.pdf")

plt.show()