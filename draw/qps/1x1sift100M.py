import numpy as np
import matplotlib.pyplot as plt
# from deep1M import naive_width4_x as naive_width4_x_d1, naive_width4_y as naive_width4_y_d1, naive_rt_width4_x as naive_rt_width4_x_d1, naive_rt_width4_y as naive_rt_width4_y_d1, pca64_width4_x as pca64_width4_x_d1, pca64_width4_y as pca64_width4_y_d1, pca64_rt_width4_x as pca64_rt_width4_x_d1, pca64_rt_width4_y as pca64_rt_width4_y_d1, ggnn_x as ggnn_x_d1, ggnn_y as ggnn_y_d1, ganns_x as ganns_x_d1, ganns_y as ganns_y_d1
# from deep10M import naive_width4_x as naive_width4_x_d10, naive_width4_y as naive_width4_y_d10, naive_rt_width4_x as naive_rt_width4_x_d10, naive_rt_width4_y as naive_rt_width4_y_d10, pca64_width4_x as pca64_width4_x_d10, pca64_width4_y as pca64_width4_y_d10, pca64_rt_width4_x as pca64_rt_width4_x_d10, pca64_rt_width4_y as pca64_rt_width4_y_d10, ggnn_x as ggnn_x_d10, ggnn_y as ggnn_y_d10, ganns_x as ganns_x_d10, ganns_y as ganns_y_d10
# from gist import naive_width4_x as naive_width4_x_g, naive_width4_y as naive_width4_y_g, naive_rt_width4_x as naive_rt_width4_x_g, naive_rt_width4_y as naive_rt_width4_y_g, pca64_width4_x as pca64_width4_x_g, pca64_width4_y as pca64_width4_y_g, pca64_rt_width4_x as pca64_rt_width4_x_g, pca64_rt_width4_y as pca64_rt_width4_y_g, ggnn_x as ggnn_x_g, ggnn_y as ggnn_y_g, ganns_x as ganns_x_g, ganns_y as ganns_y_g
# from sift1M import naive_width4_x as naive_width4_x_s1, naive_width4_y as naive_width4_y_s1, naive_rt_width4_x as naive_rt_width4_x_s1, naive_rt_width4_y as naive_rt_width4_y_s1, pca64_width4_x as pca64_width4_x_s1, pca64_width4_y as pca64_width4_y_s1, pca64_rt_width4_x as pca64_rt_width4_x_s1, pca64_rt_width4_y as pca64_rt_width4_y_s1, ggnn_x as ggnn_x_s1, ggnn_y as ggnn_y_s1, ganns_x as ganns_x_s1, ganns_y as ganns_y_s1
# from sift10M import naive_width4_x as naive_width4_x_s10, naive_width4_y as naive_width4_y_s10, naive_rt_width4_x as naive_rt_width4_x_s10, naive_rt_width4_y as naive_rt_width4_y_s10, pca64_width4_x as pca64_width4_x_s10, pca64_width4_y as pca64_width4_y_s10, pca64_rt_width4_x as pca64_rt_width4_x_s10, pca64_rt_width4_y as pca64_rt_width4_y_s10, ggnn_x as ggnn_x_s10, ggnn_y as ggnn_y_s10, ganns_x as ganns_x_s10, ganns_y as ganns_y_s10
from sift100M import pca64_width1_x as pca64_width1_x_s100, pca64_width1_y as pca64_width1_y_s100, pca64_rt_width1_x as pca64_rt_width1_x_s100, pca64_rt_width1_y as pca64_rt_width1_y_s100, pca64_width4_x as pca64_width4_x_s100, pca64_width4_y as pca64_width4_y_s100, pca64_rt_width4_x as pca64_rt_width4_x_s100, pca64_rt_width4_y as pca64_rt_width4_y_s100
# from COCO_I2I import naive_width4_x as naive_width4_x_co, naive_width4_y as naive_width4_y_co, naive_rt_width4_x as naive_rt_width4_x_co, naive_rt_width4_y as naive_rt_width4_y_co, pca64_width4_x as pca64_width4_x_co, pca64_width4_y as pca64_width4_y_co, pca64_rt_width4_x as pca64_rt_width4_x_co, pca64_rt_width4_y as pca64_rt_width4_y_co, ggnn_x as ggnn_x_co, ggnn_y as ggnn_y_co, ganns_x as ganns_x_co, ganns_y as ganns_y_co

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

# 数据集名称
# datasets = ['DEEP-1M', 'DEEP-10M', 'GIST', 'SIFT-1M', 'SIFT-10M', 'COCO-I2I']
datasets = ['SIFT100M']
methods = ['PARS', 'PARS+RT']
colors = ['#1F77B4', '#FF7F0E']
markers = ['^', 'v']

# 数据
all_groups = [
    # 第六组数据 (COCO_I2I)
    # (pca64_width1_x_s100, pca64_width1_y_s100),
    # (pca64_rt_width1_x_s100, pca64_rt_width1_y_s100),
    (pca64_width4_x_s100, pca64_width4_y_s100),
    (pca64_rt_width4_x_s100, pca64_rt_width4_y_s100),
]

# 设置颜色和标记样式

# 创建 2 行 3 列的子图布局
# fig, axs = plt.subplots(1, 1, figsize=(20, 10))
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 35})  # 所有文字统一为16号字体
sz = 35


for i, (x_data, y_data) in enumerate(all_groups[:2]):
    x_data = np.array(x_data) * 0.01
    y_data = np.array(y_data)
    y_log = np.log10(y_data)
    plt.plot(x_data, y_log, label=methods[i % len(methods)], color=colors[i % len(colors)], marker=markers[i % len(markers)], markersize=9)
    plt.xlabel('Recall', fontsize=sz)

# y
# 主刻度
main_qps = [5e5, 1e6]
main_qps_log = np.log10(main_qps)
# main_labels = [r'$5\times10^5$', r'$1\times10^6$']
main_labels = ['5e5', '1e6']
# 副刻度
minor_qps = [2e5, 3e5, 4e5, 6e5, 7e5, 8e5, 9e5, 2e6]
minor_qps_log = np.log10(minor_qps)

plt.yticks(main_qps_log, main_labels)  # 主刻度
plt.yticks(minor_qps_log, minor=True)  # 副刻度

# 控制主副刻度样式
plt.tick_params(axis='y', which='major', length=6, width=1.5)  # 主刻度
plt.tick_params(axis='y', which='minor', length=4, width=0.8)  # 副刻度

plt.tick_params(axis='both', labelsize=sz)  # ✅ 添加这行，控制坐标刻度大小

plt.title(datasets[0], fontsize=sz)
plt.ylabel('Queries per Second', fontsize=sz)
plt.grid(True, linestyle='--')
# axs[row, col].legend()


# 调整布局
plt.tight_layout()
# plt.legend(methods, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, frameon=False, fontsize=sz-5)
# plt.legend(methods, loc='lower left', fontsize=sz-5, frameon=False)
legend = plt.legend(loc='lower left', framealpha=1.0, fontsize=sz, frameon=False)
# legend.get_frame().set_facecolor('white')
# legend.get_frame().set_linewidth(1.0)
# legend.get_frame().set_boxstyle('square')

plt.show()
plt.savefig("./img/1x1.png", format="png", bbox_inches="tight")
print("图像已保存到文件：./img/1x1.png")
plt.savefig("./img/1x1.pdf", format="pdf", bbox_inches="tight")
print("图像已保存到文件：./img/1x1.pdf")