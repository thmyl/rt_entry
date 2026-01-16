#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster Reordering Script (Hilbert Edition)
功能：读取现有的 centroids 文件，利用 PCA + Hilbert Curve 
重新分配 Cluster ID，最大化 GPU Search 时的缓存命中率。
"""

import sys
import os
import struct
import numpy as np
import faiss

# 尝试导入 hilbertcurve 库
try:
    from hilbertcurve.hilbertcurve import HilbertCurve
except ImportError:
    print("错误: 缺少必要的库 'hilbertcurve'")
    print("请运行安装: pip install hilbertcurve")
    sys.exit(1)

# ==========================================
# 1. 基础工具函数 (保持不变)
# ==========================================

def read_fvecs_head(filename):
    """只读取 fvecs 头部获取 N 和 D"""
    with open(filename, 'rb') as f:
        dim_bytes = f.read(4)
        if not dim_bytes: return 0, 0
        d = struct.unpack('i', dim_bytes)[0]
        file_size = os.path.getsize(filename)
        # fvecs: d (4 bytes) + vector (d*4 bytes)
        n = file_size // (4 + d * 4)
    return n, d

def read_fbin_head(filename):
    """只读取 fbin 头部获取 N 和 D"""
    with open(filename, 'rb') as f:
        n_bytes = f.read(4)
        d_bytes = f.read(4)
        if not n_bytes or not d_bytes: return 0, 0
        n = struct.unpack('i', n_bytes)[0]
        d = struct.unpack('i', d_bytes)[0]
    return n, d

def read_centroids_file(filename, n_samples, d):
    """读取自定义的 centroids 二进制文件"""
    print(f"正在读取聚类文件: {filename} (N={n_samples}, D={d})")
    
    with open(filename, 'rb') as f:
        n_clusters = struct.unpack('i', f.read(4))[0]
        
        # 读取 Labels
        labels_bytes = f.read(n_samples * 4)
        labels = np.frombuffer(labels_bytes, dtype=np.int32).copy()
        
        # 读取 Centroids
        centroids_bytes = f.read(n_clusters * d * 4)
        centroids = np.frombuffer(centroids_bytes, dtype=np.float32).reshape(n_clusters, d)
        
        # 读取 Cluster Points 信息
        cluster_points = []
        for _ in range(n_clusters):
            count_bytes = f.read(4)
            count = struct.unpack('i', count_bytes)[0]
            if count > 0:
                p_bytes = f.read(count * 4)
                p_ids = np.frombuffer(p_bytes, dtype=np.int32)
                cluster_points.append(p_ids)
            else:
                cluster_points.append(np.array([], dtype=np.int32))
                
    return n_clusters, labels, centroids, cluster_points

def write_centroids(filename, n_clusters, labels, centroids, cluster_points):
    """写入重排后的文件"""
    print(f"正在写入重排后的文件: {filename}")
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', n_clusters))
        f.write(labels.tobytes())
        f.write(centroids.tobytes())
        for i in range(n_clusters):
            p_ids = cluster_points[i]
            f.write(struct.pack('i', len(p_ids)))
            f.write(p_ids.tobytes())

# ==========================================
# 2. Hilbert 排序核心逻辑
# ==========================================

def get_reorder_mapping(centroids):
    """
    计算 centroids 的 Hilbert 重排映射
    """
    K, d = centroids.shape
    print("Step 1: 正在进行 PCA 降维 (D -> 3)...")
    
    # 1. PCA 降维到 3维
    # 为什么要降维？因为 Hilbert 曲线在高维空间(>3)计算开销极大且稀疏，
    # 3D 已经足以捕捉主要的空间拓扑关系。
    mat = faiss.PCAMatrix(d, 3)
    mat.train(centroids)
    centroids_3d = mat.apply_py(centroids)
    
    # 2. 归一化到整数空间
    # Hilbert Curve 需要整数坐标。我们使用 10 bits 精度 (0 ~ 1023)
    # 这对于几十万个聚类中心来说足够了
    p_bits = 10 
    c_min = centroids_3d.min(axis=0)
    c_max = centroids_3d.max(axis=0)
    
    scale = (2**p_bits - 1) / (c_max - c_min + 1e-6)
    coords_int = ((centroids_3d - c_min) * scale).astype(np.int64)
    
    # 3. 计算 Hilbert Distance
    print(f"Step 2: 计算 Hilbert Curve 距离 (Bits={p_bits}, Dim=3)...")
    hc = HilbertCurve(p=p_bits, n=3)
    
    keys = []
    # 批量计算可能会爆内存或不支持，这里简单循环，速度很快 (10万级 < 1秒)
    # .tolist() 是为了兼容库的输入格式
    coords_list = coords_int.tolist()
    
    # 使用 distance_from_coordinates 计算每个点在曲线上的位置
    for i, coord in enumerate(coords_list):
        dist = hc.distance_from_point(coord)
        keys.append(dist)
    
    keys = np.array(keys, dtype=np.object_) # 使用 object 防止 overflow (虽然通常不会)
    
    # 4. 获取排序索引
    print("Step 3: 生成映射表...")
    sort_idx = np.argsort(keys) # 得到使得 keys 有序的索引数组
    
    # map_old_to_new[old_id] = new_id
    map_old_to_new = np.zeros(K, dtype=np.int32)
    for new_id, old_id in enumerate(sort_idx):
        map_old_to_new[old_id] = new_id
        
    return sort_idx, map_old_to_new

# ==========================================
# 3. 主程序
# ==========================================

def main():
    # if len(sys.argv) < 3:
    #     print("用法: python reorder_centroids_hilbert.py <dataset_path> <centroids_path>")
    #     print("示例: python reorder_centroids_hilbert.py data/base.fbin data/centroids_1024")
    #     sys.exit(1)

    # dataset_path = sys.argv[1]
    # centroids_path = sys.argv[2]

    # dataset_path = "/mnt/IntelP5520_8T_1/myl/sift100M/sift100M_base.fbin"
    # centroids_path = "/mnt/IntelP5520_8T_1/myl/cache_search/data/sift100M/centroids_5000"

    dataset_path = "/mnt/IntelP5520_8T_1/myl/deep100M/fbin/deep100M_base.fbin"
    centroids_path = "/mnt/IntelP5520_8T_1/myl/cache_search/data/deep100M/centroids_10000"
    
    if not os.path.exists(centroids_path):
        print(f"错误: 找不到文件 {centroids_path}")
        sys.exit(1)
        
    # 1. 获取元数据
    if dataset_path.endswith('.fbin'):
        n_samples, d = read_fbin_head(dataset_path)
    elif dataset_path.endswith('.fvecs'):
        n_samples, d = read_fvecs_head(dataset_path)
    else:
        print("警告: 未知后缀，尝试读取 fvecs 头部")
        n_samples, d = read_fvecs_head(dataset_path)
    
    if n_samples == 0:
        print("错误: 无法读取数据集头部信息")
        sys.exit(1)
        
    # 2. 读取旧的 centroids 文件
    K, labels, centroids, cluster_points = read_centroids_file(centroids_path, n_samples, d)
    
    # 3. 计算重排映射 (Hilbert)
    sort_idx, map_old_to_new = get_reorder_mapping(centroids)
    
    # 4. 应用重排
    print("Step 4: 正在应用重排数据...")
    
    # 4.1 重排 Centroids (物理位置改变)
    new_centroids = centroids[sort_idx]
    
    # 4.2 重排 Cluster Points (列表顺序改变)
    new_cluster_points = [cluster_points[old_id] for old_id in sort_idx]
    
    # 4.3 更新 Labels (数值改变)
    new_labels = map_old_to_new[labels]
    
    # 5. 保存结果
    dir_name = os.path.dirname(centroids_path)
    base_name = os.path.basename(centroids_path)
    # 文件名加上 _hilbert 后缀
    new_filename = os.path.join(dir_name, f"{base_name}_hilbert")
    
    write_centroids(new_filename, K, new_labels, new_centroids, new_cluster_points)
    
    print("="*50)
    print(f"处理完成！")
    print(f"输出文件: {new_filename}")
    print("-" * 50)
    print("【重要提示】C++ 代码修改指南:")
    print("1. 请加载此新生成的聚类文件。")
    print("2. 在 build_query_batches 函数中，按照 cluster ID (从小到大) 排序 queries。")
    print("3. ***必须移除*** 所有的 '奇偶交错 batch' (target_batch_indices) 逻辑。")
    print("4. 直接使用连续的 batch 索引，即可获得最佳的 L2 Cache 性能。")
    print("="*50)

if __name__ == "__main__":
    main()