import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

# 所有数据集
datasets = {
    "SIFT1M": {
        "y_values": [90, 60, 48, 38, 28, 24],
        "top_line": 128,
        # "title": "SIFT-1M"
    },
    "DEEP1M": {
        "y_values": [84, 69, 57, 49, 42, 33],
        "top_line": 96,
        # "title": "DEEP-1M"
    },
    "GIST": {
        "y_values": [378, 208, 131, 93, 61, 48],
        "top_line": 960,
        # "title": "GIST"
    }
}

x_labels = [16, 32, 64, 128, 256, 512]
bar_color = "#4C72B0"
hatch_pattern = "/"

# 设置图像大小和子图
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
sz = 35
plt.rcParams.update({'font.size': 35})

# 为图例收集所有 legend handles
handles = []

for idx, (ax, (title, data)) in enumerate(zip(axes, datasets.items())):
    y_values = data["y_values"]
    top = data["top_line"]

    # 绘制柱状图
    bars = ax.bar(range(len(x_labels)), y_values, color=bar_color, width=0.6,
                  hatch=hatch_pattern, edgecolor='black', linewidth=2)

    # 横轴刻度
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)

    # 放大坐标字体
    ax.tick_params(axis='y', labelsize=sz)
    ax.tick_params(axis='x', labelsize=sz)

    # 添加标签
    ax.set_xlabel("$n_c$", fontsize=sz)
    ax.set_title(title)

    # 添加红色虚线，并记录 legend handle
    # line = ax.axhline(y=top, color='red', linestyle='--', linewidth=2, label='Original dimension')
    # if idx == 0:
    #     handles.append(line)
    # 添加红色虚线
    line = ax.axhline(y=top, color='red', linestyle='--', linewidth=2, label='Original dimension')
    if idx == 0:
        handles.append(line)

    ax.text(len(x_labels) - 0.5, top - top * 0.02, str(top), color='red', fontsize=sz,
        ha='right', va='top', fontweight='bold')

    # 网格
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# 左边第一个图添加 y 轴标签
axes[0].set_ylabel("dimension", fontsize=sz)

# 添加图例到图像正上方，居中
fig.legend(handles=handles, labels=["Original dimension"], loc='upper center', ncol=1,
           bbox_to_anchor=(0.5, 1.02), fontsize=sz, frameon=False)

# 自动调整子图间距
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 留出上方图例空间
plt.subplots_adjust(left=0.05, right=0.98, top=0.72, bottom=0.1, wspace=0.2)


# 保存图像
plt.savefig("bar_chart_all_datasets.png", format="png", bbox_inches="tight")
# plt.savefig("bar_chart_all_datasets.pdf", format="pdf", bbox_inches="tight")
# plt.savefig("bar_chart_all_datasets_hd.png", format="png", dpi=300, bbox_inches="tight")
# print("图像已保存为 bar_chart_all_datasets.[png|pdf|hd.png]")
plt.savefig("dimension.pdf", format="pdf", dpi=300, bbox_inches="tight")
print("图像已保存为 dimension.pdf")
