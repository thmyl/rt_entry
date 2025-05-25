import numpy as np
import matplotlib.pyplot as plt
from deep1M import naive_width4_x as naive_width4_x_d1, naive_width4_y as naive_width4_y_d1, naive_rt_width4_x as naive_rt_width4_x_d1, naive_rt_width4_y as naive_rt_width4_y_d1, pca64_width4_x as pca64_width4_x_d1, pca64_width4_y as pca64_width4_y_d1, pca64_rt_width4_x as pca64_rt_width4_x_d1, pca64_rt_width4_y as pca64_rt_width4_y_d1, ggnn_x as ggnn_x_d1, ggnn_y as ggnn_y_d1, ganns_x as ganns_x_d1, ganns_y as ganns_y_d1
from deep10M import naive_width4_x as naive_width4_x_d10, naive_width4_y as naive_width4_y_d10, naive_rt_width4_x as naive_rt_width4_x_d10, naive_rt_width4_y as naive_rt_width4_y_d10, pca64_width4_x as pca64_width4_x_d10, pca64_width4_y as pca64_width4_y_d10, pca64_rt_width4_x as pca64_rt_width4_x_d10, pca64_rt_width4_y as pca64_rt_width4_y_d10, ggnn_x as ggnn_x_d10, ggnn_y as ggnn_y_d10, ganns_x as ganns_x_d10, ganns_y as ganns_y_d10
from gist import naive_width4_x as naive_width4_x_g, naive_width4_y as naive_width4_y_g, naive_rt_width4_x as naive_rt_width4_x_g, naive_rt_width4_y as naive_rt_width4_y_g, pca64_width4_x as pca64_width4_x_g, pca64_width4_y as pca64_width4_y_g, pca64_rt_width4_x as pca64_rt_width4_x_g, pca64_rt_width4_y as pca64_rt_width4_y_g, ggnn_x as ggnn_x_g, ggnn_y as ggnn_y_g, ganns_x as ganns_x_g, ganns_y as ganns_y_g
from sift1M import naive_width4_x as naive_width4_x_s1, naive_width4_y as naive_width4_y_s1, naive_rt_width4_x as naive_rt_width4_x_s1, naive_rt_width4_y as naive_rt_width4_y_s1, pca64_width4_x as pca64_width4_x_s1, pca64_width4_y as pca64_width4_y_s1, pca64_rt_width4_x as pca64_rt_width4_x_s1, pca64_rt_width4_y as pca64_rt_width4_y_s1, ggnn_x as ggnn_x_s1, ggnn_y as ggnn_y_s1, ganns_x as ganns_x_s1, ganns_y as ganns_y_s1
from sift10M import naive_width4_x as naive_width4_x_s10, naive_width4_y as naive_width4_y_s10, naive_rt_width4_x as naive_rt_width4_x_s10, naive_rt_width4_y as naive_rt_width4_y_s10, pca64_width4_x as pca64_width4_x_s10, pca64_width4_y as pca64_width4_y_s10, pca64_rt_width4_x as pca64_rt_width4_x_s10, pca64_rt_width4_y as pca64_rt_width4_y_s10, ggnn_x as ggnn_x_s10, ggnn_y as ggnn_y_s10, ganns_x as ganns_x_s10, ganns_y as ganns_y_s10
# from sift100M import pca64_width1_x as pca64_width1_x_s100, pca64_width1_y as pca64_width1_y_s100, pca64_rt_width1_x as pca64_rt_width1_x_s100, pca64_rt_width1_y as pca64_rt_width1_y_s100, pca64_width4_x as pca64_width4_x_s100, pca64_width4_y as pca64_width4_y_s100, pca64_rt_width4_x as pca64_rt_width4_x_s100, pca64_rt_width4_y as pca64_rt_width4_y_s100
from COCO_I2I import naive_width4_x as naive_width4_x_co, naive_width4_y as naive_width4_y_co, naive_rt_width4_x as naive_rt_width4_x_co, naive_rt_width4_y as naive_rt_width4_y_co, pca64_width4_x as pca64_width4_x_co, pca64_width4_y as pca64_width4_y_co, pca64_rt_width4_x as pca64_rt_width4_x_co, pca64_rt_width4_y as pca64_rt_width4_y_co, ggnn_x as ggnn_x_co, ggnn_y as ggnn_y_co, ganns_x as ganns_x_co, ganns_y as ganns_y_co

plt.rcParams['font.family'] = 'Times New Roman'

# 数据集名称
datasets = ['DEEP-1M', 'DEEP-10M', 'GIST', 'SIFT-1M', 'SIFT-10M', 'COCO-I2I']
methods = ['PCSearch', 'PCSearch+RT', 'GAGRA', 'CAGRA+RT', 'GGNN', 'GANNS']
colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
markers = ['^', 'v', 'o', 's', 'D', '*']

# 数据
all_groups = [
    # 第一组数据 (d1)
    (pca64_width4_x_d1, pca64_width4_y_d1),
    (pca64_rt_width4_x_d1, pca64_rt_width4_y_d1),
    (naive_width4_x_d1, naive_width4_y_d1),
    (naive_rt_width4_x_d1, naive_rt_width4_y_d1),
    (ggnn_x_d1, ggnn_y_d1),
    (ganns_x_d1, ganns_y_d1),
    
    # 第二组数据 (d10)
    (pca64_width4_x_d10, pca64_width4_y_d10),
    (pca64_rt_width4_x_d10, pca64_rt_width4_y_d10),
    (naive_width4_x_d10, naive_width4_y_d10),
    (naive_rt_width4_x_d10, naive_rt_width4_y_d10),
    (ggnn_x_d10, ggnn_y_d10),
    (ganns_x_d10, ganns_y_d10),

    # 第三组数据 (GIST)
    (pca64_width4_x_g, pca64_width4_y_g),
    (pca64_rt_width4_x_g, pca64_rt_width4_y_g),
    (naive_width4_x_g, naive_width4_y_g),
    (naive_rt_width4_x_g, naive_rt_width4_y_g),
    (ggnn_x_g, ggnn_y_g),
    (ganns_x_g, ganns_y_g),

    # 第四组数据 (SIFT1M)
    (pca64_width4_x_s1, pca64_width4_y_s1),
    (pca64_rt_width4_x_s1, pca64_rt_width4_y_s1),
    (naive_width4_x_s1, naive_width4_y_s1),
    (naive_rt_width4_x_s1, naive_rt_width4_y_s1),
    (ggnn_x_s1, ggnn_y_s1),
    (ganns_x_s1, ganns_y_s1),

    # 第五组数据 (SIFT10M)
    (pca64_width4_x_s10, pca64_width4_y_s10),
    (pca64_rt_width4_x_s10, pca64_rt_width4_y_s10),
    (naive_width4_x_s10, naive_width4_y_s10),
    (naive_rt_width4_x_s10, naive_rt_width4_y_s10),
    (ggnn_x_s10, ggnn_y_s10),
    (ganns_x_s10, ganns_y_s10),

    # 第六组数据 (COCO_I2I)
    # (pca64_width1_x_s100, pca64_width1_y_s100),
    # (pca64_rt_width1_x_s100, pca64_rt_width1_y_s100),
    # (pca64_width4_x_s100, pca64_width4_y_s100),
    # (pca64_rt_width4_x_s100, pca64_rt_width4_y_s100),
    (pca64_width4_x_co, pca64_width4_y_co),
    (pca64_rt_width4_x_co, pca64_rt_width4_y_co),
    (naive_width4_x_co, naive_width4_y_co),
    (naive_rt_width4_x_co, naive_rt_width4_y_co),
    (ggnn_x_co, ggnn_y_co),
    (ganns_x_co, ganns_y_co),
]

# 设置颜色和标记样式
# colors = ['#38b000', '#9fa167', '#ba181b', '#fe9000', '#6a605c', '#0a2472']

# 创建 2 行 3 列的子图布局
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
plt.rcParams.update({'font.size': 30})  # 所有文字统一为16号字体
sz = 25

# 绘制每组数据

######## 0 #######################################################################################################
subplot_id = 0
for i, (x_data, y_data) in enumerate(all_groups[:6]): 
    row = subplot_id // 3  # 计算行位置
    col = subplot_id % 3   # 计算列位置
    x_data = np.array(x_data) * 0.01
    y_data = np.array(y_data)
    y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_log, label=methods[i % len(methods)], color=colors[i % len(colors)], marker=markers[i % len(markers)])

    # x
    # axs[row, col].set_xlim(0.75, 1)
    # axs[row, col].set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    axs[row, col].set_xlabel('Recall', fontsize=sz)

    # y
    # 主刻度
    main_qps = [5e5, 1e6, 5e6]
    main_qps_log = np.log10(main_qps)
    main_labels = [r'$5\times10^5$', r'$1\times10^6$', r'$5\times10^6$']
    # 副刻度
    minor_qps = [2e5, 3e5, 4e5, 6e5, 7e5, 8e5, 9e5, 2e6, 3e6, 4e6]
    minor_qps_log = np.log10(minor_qps)

    axs[row, col].set_yticks(main_qps_log)  # 主刻度
    axs[row, col].set_yticklabels(main_labels)
    axs[row, col].set_yticks(minor_qps_log, minor=True)  # 副刻度

    # 控制主副刻度样式
    axs[row, col].tick_params(axis='y', which='major', length=6, width=1.5)  # 主刻度
    axs[row, col].tick_params(axis='y', which='minor', length=4, width=0.8)  # 副刻度

    axs[row, col].tick_params(axis='both', labelsize=sz-5)  # ✅ 添加这行，控制坐标刻度大小

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    axs[row, col].set_ylabel('Queries per Second', fontsize=sz)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

####### 1 ########################################################################################################
subplot_id = subplot_id + 1
for i, (x_data, y_data) in enumerate(all_groups[6:12]):
    row = subplot_id // 3  # 计算行位置
    col = subplot_id % 3   # 计算列位置
    x_data = np.array(x_data) * 0.01
    y_data = np.array(y_data)
    y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_log, label=methods[i % len(methods)], color=colors[i % len(colors)], marker=markers[i % len(markers)])
    # x
    # axs[row, col].set_xlim(0.75, 1)
    # axs[row, col].set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    axs[row, col].set_xlabel('Recall', fontsize=sz)

    # y
    # 主刻度
    main_qps = [1e5, 5e5, 1e6]
    main_qps_log = np.log10(main_qps)
    main_labels = [r'$1\times10^5$', r'$5\times10^5$', r'$1\times10^6$']
    # 副刻度
    minor_qps = [2e5, 3e5, 4e5, 6e5, 7e5, 8e5, 9e5, 2e6]
    minor_qps_log = np.log10(minor_qps)

    axs[row, col].set_yticks(main_qps_log)  # 主刻度
    axs[row, col].set_yticklabels(main_labels)
    axs[row, col].set_yticks(minor_qps_log, minor=True)  # 副刻度

    # 控制主副刻度样式
    axs[row, col].tick_params(axis='y', which='major', length=6, width=1.5)  # 主刻度
    axs[row, col].tick_params(axis='y', which='minor', length=4, width=0.8)  # 副刻度

    axs[row, col].tick_params(axis='both', labelsize=sz-5)

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    # axs[row, col].set_ylabel('Queries per Second', fontsize=ylabel_fontsize)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

###### 2 #########################################################################################################
subplot_id = subplot_id + 1
for i, (x_data, y_data) in enumerate(all_groups[12:18]):
    row = subplot_id // 3  # 计算行位置
    col = subplot_id % 3   # 计算列位置
    x_data = np.array(x_data) * 0.01
    y_data = np.array(y_data)
    y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_log, label=methods[i % len(methods)], color=colors[i % len(colors)], marker=markers[i % len(markers)])
    # x
    # axs[row, col].set_xlim(0.75, 1)
    # axs[row, col].set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    axs[row, col].set_xlabel('Recall', fontsize=sz)

    # y
    # 主刻度
    main_qps = [1e4, 1e5, 1e6]
    main_qps_log = np.log10(main_qps)
    main_labels = [r'$1\times10^4$', r'$1\times10^5$', r'$1\times10^6$']
    # 副刻度
    minor_qps = [2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6]
    minor_qps_log = np.log10(minor_qps)

    axs[row, col].set_yticks(main_qps_log)  # 主刻度
    axs[row, col].set_yticklabels(main_labels)
    axs[row, col].set_yticks(minor_qps_log, minor=True)  # 副刻度

    # 控制主副刻度样式
    axs[row, col].tick_params(axis='y', which='major', length=6, width=1.5)  # 主刻度
    axs[row, col].tick_params(axis='y', which='minor', length=4, width=0.8)  # 副刻度

    axs[row, col].tick_params(axis='both', labelsize=sz-5)

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    # axs[row, col].set_ylabel('Queries per Second', fontsize=ylabel_fontsize)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

###### 3 #########################################################################################################
subplot_id = subplot_id + 1
for i, (x_data, y_data) in enumerate(all_groups[18:24]):
    row = subplot_id // 3  # 计算行位置
    col = subplot_id % 3   # 计算列位置
    x_data = np.array(x_data) * 0.01
    y_data = np.array(y_data)
    y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_log, label=methods[i % len(methods)], color=colors[i % len(colors)], marker=markers[i % len(markers)])

    # x
    # axs[row, col].set_xlim(0.75, 1)
    # axs[row, col].set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    axs[row, col].set_xlabel('Recall', fontsize=sz)

    # y
    # 主刻度
    main_qps = [5e5, 1e6, 5e6]
    main_qps_log = np.log10(main_qps)
    main_labels = [r'$5\times10^5$', r'$1\times10^6$', r'$5\times10^6$']
    # 副刻度
    minor_qps = [2e5, 3e5, 4e5, 6e5, 7e5, 8e5, 9e5, 2e6, 3e6, 4e6]
    minor_qps_log = np.log10(minor_qps)

    axs[row, col].set_yticks(main_qps_log)  # 主刻度
    axs[row, col].set_yticklabels(main_labels)
    axs[row, col].set_yticks(minor_qps_log, minor=True)  # 副刻度

    # 控制主副刻度样式
    axs[row, col].tick_params(axis='y', which='major', length=6, width=1.5)  # 主刻度
    axs[row, col].tick_params(axis='y', which='minor', length=4, width=0.8)  # 副刻度

    axs[row, col].tick_params(axis='both', labelsize=sz-5)

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    axs[row, col].set_ylabel('Queries per Second', fontsize=sz)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

####### 4 ########################################################################################################
subplot_id = subplot_id + 1
for i, (x_data, y_data) in enumerate(all_groups[24:30]):
    row = subplot_id // 3  # 计算行位置
    col = subplot_id % 3   # 计算列位置
    x_data = np.array(x_data) * 0.01
    y_data = np.array(y_data)
    y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_log, label=methods[i % len(methods)], color=colors[i % len(colors)], marker=markers[i % len(markers)])
    # x
    # axs[row, col].set_xlim(0.75, 1)
    # axs[row, col].set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    axs[row, col].set_xlabel('Recall', fontsize=sz)

    # y
    # 主刻度
    main_qps = [1e5, 5e5, 1e6]
    main_qps_log = np.log10(main_qps)
    main_labels = [r'$1\times10^5$', r'$5\times10^5$', r'$1\times10^6$']
    # 副刻度
    minor_qps = [2e5, 3e5, 4e5, 6e5, 7e5, 8e5, 9e5, 2e6, 3e6, 4e6]
    minor_qps_log = np.log10(minor_qps)

    axs[row, col].set_yticks(main_qps_log)  # 主刻度
    axs[row, col].set_yticklabels(main_labels)
    axs[row, col].set_yticks(minor_qps_log, minor=True)  # 副刻度

    # 控制主副刻度样式
    axs[row, col].tick_params(axis='y', which='major', length=6, width=1.5)  # 主刻度
    axs[row, col].tick_params(axis='y', which='minor', length=4, width=0.8)  # 副刻度

    axs[row, col].tick_params(axis='both', labelsize=sz-5)

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    # axs[row, col].set_ylabel('Queries per Second', fontsize=ylabel_fontsize)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

####### 5 ########################################################################################################
subplot_id = subplot_id + 1
for i, (x_data, y_data) in enumerate(all_groups[30:36]):
    row = subplot_id // 3  # 计算行位置
    col = subplot_id % 3   # 计算列位置
    x_data = np.array(x_data) * 0.01
    y_data = np.array(y_data)
    y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_log, label=methods[i % len(methods)], color=colors[i % len(colors)], marker=markers[i % len(markers)])
    # x
    # axs[row, col].set_xlim(0.75, 1)
    # axs[row, col].set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    axs[row, col].set_xlabel('Recall', fontsize=sz)

    # y
    # 主刻度
    main_qps = [5e5, 1e6, 5e6]
    main_qps_log = np.log10(main_qps)
    main_labels = [r'$5\times10^5$', r'$1\times10^6$', r'$5\times10^6$']
    # 副刻度
    minor_qps = [2e5, 3e5, 4e5, 6e5, 7e5, 8e5, 9e5, 2e6, 3e6, 4e6]
    minor_qps_log = np.log10(minor_qps)

    axs[row, col].set_yticks(main_qps_log)  # 主刻度
    axs[row, col].set_yticklabels(main_labels)
    axs[row, col].set_yticks(minor_qps_log, minor=True)  # 副刻度

    # 控制主副刻度样式
    axs[row, col].tick_params(axis='y', which='major', length=6, width=1.5)  # 主刻度
    axs[row, col].tick_params(axis='y', which='minor', length=4, width=0.8)  # 副刻度

    axs[row, col].tick_params(axis='both', labelsize=sz-5)

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    # axs[row, col].set_ylabel('Queries per Second', fontsize=ylabel_fontsize)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

# 调整布局
plt.tight_layout()
fig.legend(methods, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, frameon=False, fontsize=sz-5)
plt.show()
plt.savefig("./img/2x3.png", format="png", bbox_inches="tight")
print("图像已保存到文件：./img/2x3.png")
plt.savefig("./img/2x3.pdf", format="pdf", bbox_inches="tight")
print("图像已保存到文件：./img/2x3.pdf")