import matplotlib.pyplot as plt

# 数据
x_labels = ['PCSearch', 'RT-PCSearch', 'CAGRA', 'GGNN', 'GANNS']
y_values = [5398, 5528, 6564, 6608, 0] 

# 设置图像大小
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 25})

# 颜色和纹理设置
bar_color = "#4C72B0"  # 顶会论文常用蓝色
hatch_pattern = "/"  # 斜线纹理

# 绘制柱状图
bars = plt.bar(range(len(x_labels)), y_values, color=bar_color, width=0.6, hatch=hatch_pattern, edgecolor='black', linewidth=2)

# 调整横轴刻度
plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=-25)

# 添加标签
# plt.xlabel("$n_c$")
plt.ylabel("MB")
plt.yscale("log", base=10)
plt.yticks([100, 1000], ["100", "1000"])

# 在柱子内部显示数值
# for bar, value in zip(bars, y_values):
#     plt.text(bar.get_x() + bar.get_width()/2, value - 5, str(value),
#              ha='center', va='top', fontsize=20, color='black')

# 显示网格
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存到文件
# plt.savefig("gist_d.pdf", format="pdf", bbox_inches="tight")
# print("图像已保存到文件：gist_d.pdf")
plt.savefig("bar_chart.png", format="png", bbox_inches="tight")
print("图像已保存到文件：bar_chart.png")

# plt.show()  # 可选，是否显示图像
