import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

# 数据集名称
datasets = ['COCO-I2I', 'crawl', 'DEEP1M', 'DEEP10M']
# colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
colors = [ '#EE822F', '#F2BA02','#4874CB', '#75BD42']
markers = ['^', 'v', 'o', 's', 'D', '*']

# pca
cocoi2i_x = np.arange(0.0006, 0.0501, 0.001)
cocoi2i_y = [0.92352, 0.92577, 0.92372, 0.92352, 0.92075, 0.92065, 0.92087, 0.92074, 0.91612, 0.91636, 0.91624, 0.91667, 0.91658, 0.91643, 0.91636, 0.91474, 0.90950, 0.90951, 0.90940, 0.90952, 0.90940, 0.90962, 0.90931, 0.90949, 0.90937, 0.90941, 0.90945, 0.90974, 0.90957, 0.90974, 0.90917, 0.90274, 0.90263, 0.90298, 0.90276, 0.90285, 0.90335, 0.90282, 0.90273, 0.90264, 0.90287, 0.90294, 0.90260, 0.90317, 0.90285, 0.90315, 0.90294, 0.90305, 0.90267, 0.90295]

crawl_x = np.arange(0.000256, 0.0501, 0.001)
crawl_y = [0.92219, 0.92076, 0.91990, 0.91955, 0.91994, 0.92023, 0.91979, 0.91978, 0.91873, 0.91838, 0.91890, 0.91829, 0.91861, 0.91875, 0.91904, 0.91867, 0.91669, 0.91661, 0.91653, 0.91690, 0.91694, 0.91653, 0.91700, 0.91644, 0.91644, 0.91681, 0.91629, 0.91689, 0.91664, 0.91661, 0.91608, 0.91548, 0.91443, 0.91395, 0.91368, 0.91396, 0.91387, 0.91399, 0.91433, 0.91398, 0.91407, 0.91413, 0.91386, 0.91416, 0.91430, 0.91373, 0.91419, 0.91425, 0.91400, 0.91381]

deep1M_x = np.arange(0.000128, 0.0501, 0.001)
deep1M_y = [0.93097, 0.92894, 0.92932, 0.92931, 0.92788, 0.92785, 0.92786, 0.92785, 0.92775, 0.92781, 0.92787, 0.92769, 0.92780, 0.92778, 0.92782, 0.92782, 0.92605, 0.92607, 0.92598, 0.92605, 0.92590, 0.92601, 0.92604, 0.92596, 0.92597, 0.92599, 0.92601, 0.92601, 0.92604, 0.92602, 0.92594, 0.92524, 0.92505, 0.92495, 0.92504, 0.92507, 0.92497, 0.92495, 0.92496, 0.92504, 0.92504, 0.92504, 0.92499, 0.92500, 0.92500, 0.92502, 0.92504, 0.92506, 0.92509, 0.92499]

deep10M_x = np.arange(0.0000512, 0.0501, 0.001)
deep10M_y = [0.92442, 0.92085, 0.91994, 0.91993, 0.92005, 0.91966, 0.91992, 0.91974, 0.91804, 0.91814, 0.91830, 0.91794, 0.91805, 0.91796, 0.91812, 0.91798, 0.91583, 0.91606, 0.91595, 0.91570, 0.91592, 0.91598, 0.91583, 0.91608, 0.91577, 0.91608, 0.91566, 0.91580, 0.91586, 0.91590, 0.91599, 0.91648, 0.91373, 0.91376, 0.91371, 0.91379, 0.91378, 0.91370, 0.91394, 0.91383, 0.91377, 0.91364, 0.91366, 0.91386, 0.91379, 0.91364, 0.91379, 0.91389, 0.91378, 0.91374, 0.91376]

# 数据
all_groups = [
    # 第一组数据
    (cocoi2i_x, cocoi2i_y),
    # 第二组数据
    (crawl_x, crawl_y),
    # 第三组数据 
    (deep1M_x, deep1M_y),
    # 第四组数据 
    (deep10M_x, deep10M_y),
]

# 创建 2 行 2 列的子图布局
fig, axs = plt.subplots(2, 2, figsize=(20, 13))
plt.rcParams.update({'font.size': 30})  # 所有文字统一为16号字体
sz = 40
line_w=3
marker_sz=8

# 绘制每组数据

######## 0 #######################################################################################################
subplot_id = 0
for i, (x_data, y_data) in enumerate(all_groups[:1]): 
    row = subplot_id // 2  # 计算行位置
    col = subplot_id % 2   # 计算列位置
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    # y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_data, color=colors[i % len(colors)], marker=markers[i % len(markers)], linewidth=line_w, markersize=marker_sz)

    # x
    axs[row, col].set_xlabel('Subdivision Ratio', fontsize=sz)

    # y
    axs[row, col].tick_params(axis='both', labelsize=sz-5)  # ✅ 添加这行，控制坐标刻度大小

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    axs[row, col].set_ylabel('Recall', fontsize=sz)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

####### 1 ########################################################################################################
subplot_id = subplot_id + 1
for i, (x_data, y_data) in enumerate(all_groups[1:2]):
    row = subplot_id // 2  # 计算行位置
    col = subplot_id % 2   # 计算列位置
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    # y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_data, color=colors[i % len(colors)], marker=markers[i % len(markers)], linewidth=line_w, markersize=marker_sz)
    # x
    axs[row, col].set_xlabel('Subdivision Ratio', fontsize=sz)

    # y
    axs[row, col].tick_params(axis='both', labelsize=sz-5)

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    # axs[row, col].set_ylabel('', fontsize=sz)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

###### 2 #########################################################################################################
subplot_id = subplot_id + 1
for i, (x_data, y_data) in enumerate(all_groups[2:3]):
    row = subplot_id // 2  # 计算行位置
    col = subplot_id % 2   # 计算列位置
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    # y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_data, color=colors[i % len(colors)], marker=markers[i % len(markers)], linewidth=line_w, markersize=marker_sz)
    # x
    axs[row, col].set_xlabel('Subdivision Ratio', fontsize=sz)

    # y
    axs[row, col].tick_params(axis='both', labelsize=sz-5)

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    axs[row, col].set_ylabel('Recall', fontsize=sz)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

###### 3 #########################################################################################################
subplot_id = subplot_id + 1
for i, (x_data, y_data) in enumerate(all_groups[3:4]):
    row = subplot_id // 2  # 计算行位置
    col = subplot_id % 2   # 计算列位置
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    # y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_data, color=colors[i % len(colors)], marker=markers[i % len(markers)], linewidth=line_w, markersize=marker_sz)

    # x
    axs[row, col].set_xlabel('Subdivision Ratio', fontsize=sz)

    # y
    axs[row, col].tick_params(axis='both', labelsize=sz-5)

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    # axs[row, col].set_ylabel('', fontsize=sz)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()


# 调整布局
plt.tight_layout()
# fig.legend(width, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, frameon=False, fontsize=sz-5)
plt.show()
plt.savefig("./img/2x2.png", format="png", bbox_inches="tight")
print("图像已保存到文件：./img/2x2.png")
plt.savefig("./img/2x2.pdf", format="pdf", bbox_inches="tight")
print("图像已保存到文件：./img/2x2.pdf")