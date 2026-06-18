import json
import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. 准备工作：设置字体和绘图风格
# -----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 30})  # 基础字体大小

colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD']
markers = ['^', 'v', 'o', 's']
label = [r'$n_c = 1000$', r'$n_c = 5000$', r'$n_c = 10000$', r'Base']

# sz = 45           
sz = 35
line_w = 3        
marker_sz = 17    

# -----------------------------------------------------------------------------
# 2. 读取数据
# -----------------------------------------------------------------------------
json_filename = '100M_compute.json'
with open(json_filename, 'r') as f:
    data = json.load(f)

# -----------------------------------------------------------------------------
# 3. 开始绘图
# -----------------------------------------------------------------------------
# fig, axs = plt.subplots(1, 2, figsize=(22, 6))
fig, axs = plt.subplots(1, 2, figsize=(22, 8))

keys_to_plot = ["1000", "5000", "10000", "base"]
labels_map = {
    "1000": "Top-1000",
    "5000": "Top-5000",
    "10000": "Top-10000",
    "base": "Base"
}

# ----------------- 左图: Deep100M -----------------
ax_deep = axs[0]
dataset_deep = data.get("deep100M_compute", {})

for i, key in enumerate(keys_to_plot):
    if key in dataset_deep:
        xy_data = dataset_deep[key]
        
        # 1. 先转为 numpy 数组，方便筛选
        raw_mem = np.array(xy_data['memory'])
        raw_comp = np.array(xy_data['compute'])
        
        # 2. 如果是 base，进行筛选
        if key == "base":
            mask = raw_mem > 39212  # 创建布尔掩码
            raw_mem = raw_mem[mask] # 筛选 memory
            raw_comp = raw_comp[mask] # 筛选对应的 compute
        
        # 3. 计算最终坐标 (转换单位)
        x_vals = raw_mem / 1024
        y_vals = raw_comp
        
        ax_deep.plot(x_vals, y_vals, 
                     color=colors[i % len(colors)], 
                     marker=markers[i % len(markers)], 
                     linewidth=line_w, 
                     markersize=marker_sz, 
                     label=labels_map[key])

ax_deep.set_title("DEEP100M", fontsize=sz, pad=15)
ax_deep.set_xlabel('Memory (GB)', fontsize=sz)
ax_deep.set_ylabel('Dimension', fontsize=sz)
ax_deep.tick_params(axis='both', labelsize=sz-5)
ax_deep.grid(True, linestyle='--')

# legend = ax_deep.legend(framealpha=1.0, fontsize=20)
# legend.get_frame().set_facecolor('white')
# legend.get_frame().set_boxstyle('square')


# ----------------- 右图: Sift100M -----------------
ax_sift = axs[1]
dataset_sift = data.get("sift100M_compute", {})

for i, key in enumerate(keys_to_plot):
    if key in dataset_sift:
        xy_data = dataset_sift[key]
        
        # 1. 先转为 numpy 数组
        raw_mem = np.array(xy_data['memory'])
        raw_comp = np.array(xy_data['compute'])
        
        # 2. 如果是 base，进行筛选
        if key == "base":
            mask = raw_mem > 39212
            raw_mem = raw_mem[mask]
            raw_comp = raw_comp[mask]
            
        # 3. 计算最终坐标
        x_vals = raw_mem / 1024
        y_vals = raw_comp
        
        ax_sift.plot(x_vals, y_vals, 
                     color=colors[i % len(colors)], 
                     marker=markers[i % len(markers)], 
                     linewidth=line_w, 
                     markersize=marker_sz, 
                     label=labels_map[key])

ax_sift.set_title("SIFT100M", fontsize=sz, pad=15)
ax_sift.set_xlabel('Memory (GB)', fontsize=sz)
# ax_sift.set_ylabel('Compute', fontsize=sz)
ax_sift.tick_params(axis='both', labelsize=sz)
ax_sift.grid(True, linestyle='--')

# legend = ax_sift.legend(framealpha=1.0, fontsize=20)
# legend.get_frame().set_facecolor('white')
# legend.get_frame().set_boxstyle('square')


# -----------------------------------------------------------------------------
# 4. 保存文件
# -----------------------------------------------------------------------------
plt.tight_layout()
fig.legend(label, loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=6, frameon=False, fontsize=sz) # bbox_to_anchor=(x, y) y越大，图例越往上跑

output_dir = "./img"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

png_path = os.path.join(output_dir, "1x2.png")
plt.savefig(png_path, format="png", bbox_inches="tight")
print(f"图像已保存到文件：{png_path}")

pdf_path = os.path.join(output_dir, "1x2.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
print(f"图像已保存到文件：{pdf_path}")

plt.show()