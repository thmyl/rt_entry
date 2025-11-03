import matplotlib.pyplot as plt

# 数据
labels = ['PARS', 'PARS+RT']
values = [37140, 37676]
values = [v / 1024 for v in values]  # 转换为GB
colors = ['#1F77B4', '#FF7F0E']  # 顶会论文常用蓝色和绿色
hatch = ['/', '\\']

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 25})

# 创建柱状图
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(labels, values, color=colors, hatch=hatch, width=0.6)
ax.set_ylim(0, max(values) + 5)

# 添加柱子上的数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=22)

# 设置坐标轴标签和标题（可根据需要修改）
ax.set_ylabel('Memory Usage (GB)')

plt.tight_layout()
plt.show()


fig.savefig("sift100m_memory.png", format="png",bbox_inches="tight")
fig.savefig("sift100m_memory.pdf", format="pdf", bbox_inches="tight")
print("图像已保存到文件：sift100m_memory.(png/pdf)")

# 显示图像
plt.show()
