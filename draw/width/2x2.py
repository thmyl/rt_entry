import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

# 数据集名称
datasets = ['SIFT-1M (PCSearch)', 'COCO-I2I (PCSearch)', 'SIFT-1M (PCSearch+RT)', 'COCO-I2I (PCSearch+RT)']
width = ['w=1', 'w=2', 'w=4', 'w=8']
# colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
colors = [ '#EE822F', '#F2BA02','#4874CB', '#75BD42']
markers = ['^', 'v', 'o', 's', 'D', '*']

# pca
sift1M_w1_x = [60.44, 82.58, 90.23, 93.89, 95.95, 97.21, 97.97, 98.45, 98.79, 99.05, 99.24, 99.37]
sift1M_w1_y = [7328638.119, 5060959.254, 3883857.136, 3117090.383, 2630969.67, 2258228.42, 1986957.61, 1772892.474, 1596546.351, 1448553.185, 1322114.749, 1225650.913]
sift1M_w2_x = [79.38, 87.84, 91.97, 94.40, 95.97, 96.97, 97.68, 98.15, 98.51, 98.78, 99.00, 99.18]
sift1M_w2_y = [4884911.485, 4025975.595, 3418604.731, 2986679.41, 2628915.441, 2355252.094, 2124080.273, 1940782.834, 1774289.131, 1636438.913, 1524624.98, 1428330.653]
sift1M_w4_x = [76.42, 86.87, 91.55, 94.21, 95.87, 96.95, 97.67, 98.16, 98.53, 98.79, 98.99, 99.17]
sift1M_w4_y = [4662874.196, 3943715.295, 3379622.902, 2974623.487, 2622132.37, 2364082.96, 2140136.113, 1955883.101, 1792606.216, 1665575.715, 1544847.701, 1443418.014]
sift1M_w8_x = [81.10, 89.29, 93.00, 95.06, 96.53, 97.38, 98.02, 98.39, 98.71, 98.94, 99.11, 99.24]
sift1M_w8_y = [3132753.565, 2757183.151, 2484256.027, 2265857.604, 2063906.81, 1877264.45, 1741550.433, 1604178.564, 1516704.989, 1414809.376, 1338930.061, 1261512.882]


cocoi2i_w1_x = [63.96, 71.88, 77.34, 81.43, 84.36, 86.82, 88.65, 90.66, 92.01, 92.82, 93.43, 94.26, 95.22, 95.96, 96.65, 97.20, 97.49, 97.57]
cocoi2i_w1_y = [ 11691940.75, 10652009.5, 9911981.603, 9126585.744, 8407600.471, 7966794.401, 7499175.091, 6869924.843, 6366142.309, 6052426.115, 5777775.21, 5374379.259, 4858118.645, 4406625.803, 3926850.627, 3467105.833, 3122989.575, 3109017.706 ]
cocoi2i_w2_x = [ 46.67, 59.33, 68.88, 75.54, 80.32, 83.82, 86.35, 88.42, 91.21, 92.94, 94.13, 95.00, 96.41, 97.12, 97.54, 97.62, 97.63, 97.64 ]
cocoi2i_w2_y = [ 11717148.05, 10838111.13, 10208352.47, 9807960.14, 9222369.78, 8775470.804, 8331042.297, 7968952.959, 7316252.323, 6721378.689, 6259271.546, 5828626.717, 4961843.424, 4318180.837, 3898118.768, 3819870.201, 3759751.856, 3707218.696 ]
cocoi2i_w4_x = [ 66.00, 78.52, 85.72, 89.75, 92.21, 93.71, 94.76, 95.52, 96.53, 97.10, 97.48, 97.64, 97.71 ]
cocoi2i_w4_y = [ 10489877.27, 9460737.938, 8560984.171, 8126777.733, 7501931.747, 7071785.697, 6621595.672, 6316720.359, 5564582.545, 5064060.364, 4625411.083, 4465362.186, 4340598.048 ]
cocoi2i_w8_x = [8.93, 31.41, 58.96, 80.88, 90.40, 94.26, 95.96, 96.81, 97.30, 97.59, 97.71, 97.79, 97.80, 97.80, 97.80, 97.80, 97.80, 97.79 ]
cocoi2i_w8_y = [ 10082068.03, 8421549.06, 7376045.554, 6488197.968, 5792031.323, 5283178.36, 4868975.86, 4511210.358, 4169846.174, 3974704.978, 3868711.41, 3822513.073, 3791124.22, 3721594.927, 3708084.737, 3704787.697, 3688988.738, 3711897.373 ]

# pca+rt
sift1M_w1_rt_x = [67.07, 84.43, 90.95, 94.23, 96.15, 97.30, 98.02, 98.47, 98.81, 99.07, 99.23, 99.38]
sift1M_w1_rt_y = [6927270.586, 4783887.866, 3775921.702, 3074189.413, 2581704.493, 2227315.35, 1935351.518, 1719634.268, 1556863.667, 1416256.07, 1309567.306, 1211249.113]
sift1M_w2_rt_x = [83.23, 89.51, 92.90, 94.98, 96.32, 97.24, 97.86, 98.28, 98.62, 98.86, 99.06, 99.21]
sift1M_w2_rt_y = [4125974.246, 3538219.851, 3027055.825, 2674583.434, 2377685.893, 2142001.872, 1958863.859, 1796557.795, 1668527.074, 1535687.85, 1438424.637, 1341664.173]
sift1M_w4_rt_x = [84.25, 90.07, 93.30, 95.19, 96.51, 97.40, 97.97, 98.37, 98.68, 98.92, 99.09, 99.23]
sift1M_w4_rt_y = [4468714.53, 3882982.441, 3279667.048, 2868189.484, 2574983.52, 2293278.172, 2098204.357, 1928930.485, 1750023.188, 1628847.541, 1516803.912, 1421522.707]
sift1M_w8_rt_x = [89.72, 93.01, 95.10, 96.42, 97.40, 97.94, 98.36, 98.68, 98.93, 99.10, 99.23, 99.34]
sift1M_w8_rt_y = [2836259.881, 2515122.172, 2311513.417, 2079477.635, 1918340.099, 1760671.872, 1628131.507, 1540665.876, 1429004.213, 1344073.376, 1268010.504, 1200738.214]


cocoi2i_w1_rt_x = [76.85, 81.28, 84.38, 86.83, 88.70, 90.25, 91.41, 92.77, 93.79, 94.28, 94.71, 95.32, 96.07, 96.60, 97.06, 97.49, 97.68, 97.69]
cocoi2i_w1_rt_y = [10297494.62, 9615384.615, 9100837.277, 8438391.305, 8000000, 7393496.68, 6959765.595, 6369426.752, 5916214.569, 5677108.762, 5434782.609, 5101624.357, 4558113.67, 4201292.318, 3752993.012, 3324070.258, 3099026.286, 3077718.549]
cocoi2i_w2_rt_x = [71.86, 77.93, 82.17, 85.21, 87.52, 89.26, 90.68, 91.76, 93.43, 94.49, 95.33, 95.95, 96.96, 97.47, 97.71, 97.74, 97.74, 97.74]
cocoi2i_w2_rt_y = [10590303.52, 9803921.569, 9279020.135, 8928571.429, 8642892.949, 8147635.149, 7751937.984, 7407407.407, 6891703.767, 6342601.989, 5817640.249, 5405405.405, 4694835.681, 4131326.61, 3780675.455, 3790721.829, 3768011.093, 3725657.485]
cocoi2i_w4_rt_x = [45.16, 63.78, 74.86, 85.03, 89.36, 91.94, 93.61, 94.66, 95.45, 96.08, 96.53, 97.13, 97.54, 97.74, 97.79, 97.81]
cocoi2i_w4_rt_y = [12658708.56, 11484484.46, 10579323.77, 9399821.403, 8745539.775, 8002112.558, 7622474.103, 7219851.704, 6679758.994, 6309944.472, 5994952.25, 5203698.789, 4859039.271, 4636777.996, 4528923.973, 4399646.268]
cocoi2i_w8_rt_x = [53.75, 74.43, 84.00, 92.02, 94.85, 96.29, 97.04, 97.48, 97.64, 97.71, 97.75, 97.77, 97.79, 97.78, 97.79, 97.79, 97.78, 97.79]
cocoi2i_w8_rt_y = [9380159.087, 7526040.099, 7174733.459, 6312932.041, 5648026.297, 5170122.894, 4716981.132, 4443279.318, 4249388.088, 4185168.6, 4048894.449, 3995956.092, 4076873.527, 4087221.303, 3981604.985, 4083282.633, 4119396.591, 3998064.937]


# 数据
all_groups = [
    # 第一组数据
    (sift1M_w1_x, sift1M_w1_y),
    (sift1M_w2_x, sift1M_w2_y),
    (sift1M_w4_x, sift1M_w4_y),
    (sift1M_w8_x, sift1M_w8_y),
    
    # 第二组数据
    (cocoi2i_w1_x, cocoi2i_w1_y),
    (cocoi2i_w2_x, cocoi2i_w2_y),
    (cocoi2i_w4_x, cocoi2i_w4_y),
    (cocoi2i_w8_x, cocoi2i_w8_y),

    # 第三组数据 
    (sift1M_w1_rt_x, sift1M_w1_rt_y),
    (sift1M_w2_rt_x, sift1M_w2_rt_y),
    (sift1M_w4_rt_x, sift1M_w4_rt_y),
    (sift1M_w8_rt_x, sift1M_w8_rt_y),

    # 第四组数据 
    (cocoi2i_w1_rt_x, cocoi2i_w1_rt_y),
    (cocoi2i_w2_rt_x, cocoi2i_w2_rt_y),
    (cocoi2i_w4_rt_x, cocoi2i_w4_rt_y),
    (cocoi2i_w8_rt_x, cocoi2i_w8_rt_y),

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
for i, (x_data, y_data) in enumerate(all_groups[:4]): 
    row = subplot_id // 2  # 计算行位置
    col = subplot_id % 2   # 计算列位置
    x_data = np.array(x_data) * 0.01
    y_data = np.array(y_data)
    y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_log, label=width[i % len(width)], color=colors[i % len(colors)], marker=markers[i % len(markers)], linewidth=line_w, markersize=marker_sz)

    # x
    axs[row, col].set_xlim(0.75, 1)
    axs[row, col].set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    axs[row, col].set_xlabel('Recall', fontsize=sz)

    # y
    # 主刻度
    main_qps = [1e6, 5e6]
    main_qps_log = np.log10(main_qps)
    main_labels = [r'$1\times10^6$', r'$5\times10^6$']
    # 副刻度
    minor_qps = [2e6, 3e6, 4e6, 7e6]
    minor_qps_log = np.log10(minor_qps)

    axs[row, col].set_yticks(main_qps_log)  # 主刻度
    axs[row, col].set_yticklabels(main_labels)
    axs[row, col].set_yticks(minor_qps_log, minor=True)  # 副刻度

    # 控制主副刻度样式
    axs[row, col].tick_params(axis='y', which='major', length=6, width=1.5)  # 主刻度
    axs[row, col].tick_params(axis='y', which='minor', length=4, width=0.8)  # 副刻度

    axs[row, col].tick_params(axis='both', labelsize=sz-5)  # ✅ 添加这行，控制坐标刻度大小

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    axs[row, col].set_ylabel('QPS', fontsize=sz)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

####### 1 ########################################################################################################
subplot_id = subplot_id + 1
for i, (x_data, y_data) in enumerate(all_groups[4:8]):
    row = subplot_id // 2  # 计算行位置
    col = subplot_id % 2   # 计算列位置
    x_data = np.array(x_data) * 0.01
    y_data = np.array(y_data)
    y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_log, label=width[i % len(width)], color=colors[i % len(colors)], marker=markers[i % len(markers)], linewidth=line_w, markersize=marker_sz)
    # x
    axs[row, col].set_xlim(0.75, 1)
    axs[row, col].set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    axs[row, col].set_xlabel('Recall', fontsize=sz)

    # y
    # 主刻度
    main_qps = [5e6, 1e7]
    main_qps_log = np.log10(main_qps)
    main_labels = [r'$5\times10^6$', r'$1\times10^7$']
    # 副刻度
    minor_qps = [3e6, 4e6, 6e6, 7e6, 8e6, 9e6]
    minor_qps_log = np.log10(minor_qps)

    axs[row, col].set_yticks(main_qps_log)  # 主刻度
    axs[row, col].set_yticklabels(main_labels)
    axs[row, col].set_yticks(minor_qps_log, minor=True)  # 副刻度

    # 控制主副刻度样式
    axs[row, col].tick_params(axis='y', which='major', length=6, width=1.5)  # 主刻度
    axs[row, col].tick_params(axis='y', which='minor', length=4, width=0.8)  # 副刻度

    axs[row, col].tick_params(axis='both', labelsize=sz-5)

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    # axs[row, col].set_ylabel('QPS', fontsize=sz)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

###### 2 #########################################################################################################
subplot_id = subplot_id + 1
for i, (x_data, y_data) in enumerate(all_groups[8:12]):
    row = subplot_id // 2  # 计算行位置
    col = subplot_id % 2   # 计算列位置
    x_data = np.array(x_data) * 0.01
    y_data = np.array(y_data)
    y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_log, label=width[i % len(width)], color=colors[i % len(colors)], marker=markers[i % len(markers)], linewidth=line_w, markersize=marker_sz)
    # x
    # axs[row, col].set_xlim(0.75, 1)
    # axs[row, col].set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    axs[row, col].set_xlim(0.85, 1)
    axs[row, col].set_xticks([0.85, 0.90, 0.95, 1.0])
    axs[row, col].set_xlabel('Recall', fontsize=sz)

    # y
    # 主刻度
    main_qps = [1e6, 5e6]
    main_qps_log = np.log10(main_qps)
    main_labels = [r'$1\times10^6$', r'$5\times10^6$']
    # 副刻度
    minor_qps = [2e6, 3e6, 4e6]
    minor_qps_log = np.log10(minor_qps)

    axs[row, col].set_yticks(main_qps_log)  # 主刻度
    axs[row, col].set_yticklabels(main_labels)
    axs[row, col].set_yticks(minor_qps_log, minor=True)  # 副刻度

    # 控制主副刻度样式
    axs[row, col].tick_params(axis='y', which='major', length=6, width=1.5)  # 主刻度
    axs[row, col].tick_params(axis='y', which='minor', length=4, width=0.8)  # 副刻度

    axs[row, col].tick_params(axis='both', labelsize=sz-5)

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    axs[row, col].set_ylabel('QPS', fontsize=sz)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()

###### 3 #########################################################################################################
subplot_id = subplot_id + 1
for i, (x_data, y_data) in enumerate(all_groups[12:16]):
    row = subplot_id // 2  # 计算行位置
    col = subplot_id % 2   # 计算列位置
    x_data = np.array(x_data) * 0.01
    y_data = np.array(y_data)
    y_log = np.log10(y_data)
    axs[row, col].plot(x_data, y_log, label=width[i % len(width)], color=colors[i % len(colors)], marker=markers[i % len(markers)], linewidth=line_w, markersize=marker_sz)

    # x
    axs[row, col].set_xlim(0.75, 1)
    axs[row, col].set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    axs[row, col].set_xlabel('Recall', fontsize=sz)

    # y
    # 主刻度
    main_qps = [5e6, 1e7]
    main_qps_log = np.log10(main_qps)
    main_labels = [r'$5\times10^6$', r'$1\times10^7$']
    # 副刻度
    minor_qps = [3e6, 4e6, 6e6, 7e6, 8e6, 9e6]
    minor_qps_log = np.log10(minor_qps)

    axs[row, col].set_yticks(main_qps_log)  # 主刻度
    axs[row, col].set_yticklabels(main_labels)
    axs[row, col].set_yticks(minor_qps_log, minor=True)  # 副刻度

    # 控制主副刻度样式
    axs[row, col].tick_params(axis='y', which='major', length=6, width=1.5)  # 主刻度
    axs[row, col].tick_params(axis='y', which='minor', length=4, width=0.8)  # 副刻度

    axs[row, col].tick_params(axis='both', labelsize=sz-5)

    axs[row, col].set_title(datasets[subplot_id], fontsize=sz)
    # axs[row, col].set_ylabel('QPS', fontsize=sz)
    axs[row, col].grid(True, linestyle='--')
    # axs[row, col].legend()


# 调整布局
plt.tight_layout()
fig.legend(width, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, frameon=False, fontsize=sz-5)
plt.show()
plt.savefig("./img/2x2.png", format="png", bbox_inches="tight")
print("图像已保存到文件：./img/2x2.png")
plt.savefig("./img/2x2.pdf", format="pdf", bbox_inches="tight")
print("图像已保存到文件：./img/2x2.pdf")