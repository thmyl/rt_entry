#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
普通 K-means 聚类程序：纯 NumPy 实现版本
特点：
1. 彻底移除 faiss, sklearn, scipy 依赖，解决 threadpoolctl 报错。
2. 手写了基于 NumPy 广播机制的高效距离计算和聚类更新逻辑。
3. 保持了抽样训练和分批精确分配的功能。
"""


import os
os.environ["OMP_NUM_THREADS"] = "16" 
os.environ["OPENBLAS_NUM_THREADS"] = "16" 
os.environ["MKL_NUM_THREADS"] = "16" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" 
os.environ["NUMEXPR_NUM_THREADS"] = "16" 
import sys
import numpy as np
import struct

import time

def read_fvecs(filename):
    """读取fvecs格式文件"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes: break
            dim = struct.unpack('i', dim_bytes)[0]
            vector_bytes = f.read(dim * 4)
            if not vector_bytes: break
            vector = struct.unpack('f' * dim, vector_bytes)
            vectors.append(np.array(vector, dtype=np.float32))
    return np.array(vectors)

def read_fbin(filename):
    """读取fbin格式文件"""
    with open(filename, 'rb') as f:
        n_bytes = f.read(4)
        if not n_bytes: raise ValueError("无法读取文件头: n")
        n = struct.unpack('i', n_bytes)[0]
        d_bytes = f.read(4)
        if not d_bytes: raise ValueError("无法读取文件头: d")
        d = struct.unpack('i', d_bytes)[0]
        data_bytes = f.read(n * d * 4)
        if len(data_bytes) != n * d * 4:
            raise ValueError(f"数据不完整")
        vectors = np.frombuffer(data_bytes, dtype=np.float32).reshape(n, d)
    return vectors

def write_centroids(centroids_file, n_clusters, labels, centroids, cluster_points):
    """写入聚类结果"""
    with open(centroids_file, 'wb') as f:
        f.write(struct.pack('i', n_clusters))
        for label in labels:
            f.write(struct.pack('i', label))
        for centroid in centroids:
            for val in centroid:
                f.write(struct.pack('f', val))
        for cluster_id in range(n_clusters):
            point_ids = cluster_points[cluster_id]
            f.write(struct.pack('i', len(point_ids)))
            for point_id in point_ids:
                f.write(struct.pack('i', point_id))

def compute_distances_no_loops(data, centroids):
    """
    使用 NumPy 广播机制计算欧氏距离平方
    dist^2 = x^2 + c^2 - 2xc
    """
    # (N, 1)
    data_sq = np.sum(data**2, axis=1, keepdims=True)
    # (1, K)
    centroids_sq = np.sum(centroids**2, axis=1, keepdims=True).T
    # (N, K) = (N, 1) + (1, K) - (N, K)
    # 注意：使用 float32 可能会有微小精度误差导致负数，clip 一下
    dists = data_sq + centroids_sq - 2 * np.dot(data, centroids.T)
    return np.maximum(dists, 0)

def train_kmeans_numpy(data, K, max_iter=20):
    """
    纯 NumPy 实现的 K-means 训练
    """
    n_samples, dim = data.shape
    
    # 1. 初始化：随机选择 K 个点
    np.random.seed(42)
    random_indices = np.random.choice(n_samples, K, replace=False)
    centroids = data[random_indices].copy()
    
    for i in range(max_iter):
        # 计算距离并分配标签
        dists = compute_distances_no_loops(data, centroids)
        labels = np.argmin(dists, axis=1)
        
        # 更新聚类中心
        new_centroids = np.zeros_like(centroids)
        # 统计每个聚类的点数
        counts = np.bincount(labels, minlength=K)
        
        # 为了避免循环，使用加权求和技巧或简单的循环（K通常不大，循环K次很快）
        change = 0.0
        for k in range(K):
            if counts[k] > 0:
                # 取出属于该类的点并求均值
                new_centroids[k] = np.mean(data[labels == k], axis=0)
            else:
                # 如果某个类空了，随机重新初始化一个点
                new_centroids[k] = data[np.random.randint(n_samples)]
        
        # 检查收敛 (中心点移动距离)
        shift = np.sum((centroids - new_centroids)**2)
        centroids = new_centroids
        
        # verbose 模拟
        # print(f"  Iteration {i+1}/{max_iter}, shift: {shift:.4f}")
        if shift < 1e-6:
            break
            
    return centroids

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/cluster.py <K>")
        sys.exit(1)
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    # 配置路径
    dataset_path = "/mnt/IntelP5520_8T_1/myl/sift100M/sift100M_base.fbin"
    
    dataset_name = os.path.basename(dataset_path).split('_')[0]
    data_root = os.path.join("/mnt/IntelP5520_8T_1/myl/cache_search/data", dataset_name)
    os.makedirs(data_root, exist_ok=True)
    centroids_file = os.path.join(data_root, f"centroids_{K}")
    log_file = os.path.join(data_root, "cluster.log")
    
    def log(msg):
        print(msg)
        with open(log_file, 'a') as lf:
            lf.write(str(msg) + "\n")

    if os.path.exists(centroids_file):
        log(f"centroids_{K}文件已存在，跳过聚类计算")
        return

    log(f"开始读取数据集: {dataset_path}")
    if dataset_path.endswith('.fbin'):
        dataset = read_fbin(dataset_path)
    else:
        dataset = read_fvecs(dataset_path)
    
    n_samples = dataset.shape[0]
    log(f"数据集大小: {dataset.shape}")

    # --- 阶段1：抽样训练 ---
    log(f"第1步: 训练 K-means 聚类中心 (Pure Numpy)")
    t0 = time.time()
    
    # 抽样逻辑：最多取 25.6万 或 256*K
    target_samples = max(256000, 256 * K)
    if n_samples > target_samples:
        log(f"  - 数据集过大，抽取 {target_samples} 个样本进行训练...")
        np.random.seed(42)
        indices = np.random.choice(n_samples, target_samples, replace=False)
        train_data = dataset[indices]
    else:
        train_data = dataset

    centroids = train_kmeans_numpy(train_data, K, max_iter=20)
    log(f"K-means 训练完成，耗时 {time.time()-t0:.2f}s")

    # --- 阶段2：全量精确分配 (分批防止OOM) ---
    log(f"第2步: 计算每个点所属的最近聚类中心 (Exact Assignment)")
    t1 = time.time()
    
    final_labels = np.zeros(n_samples, dtype=np.int32)
    # 仅用于计算 avg distance
    total_dist_sq = 0.0
    
    batch_size = 10000
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_data = dataset[i:end_idx]
        
        # 使用 pure numpy 计算距离矩阵 [batch, K]
        dists_sq = compute_distances_no_loops(batch_data, centroids)
        
        # 找到最近的中心索引
        batch_labels = np.argmin(dists_sq, axis=1)
        batch_min_dists = dists_sq[np.arange(dists_sq.shape[0]), batch_labels]
        
        final_labels[i:end_idx] = batch_labels
        total_dist_sq += np.sum(batch_min_dists)
        
        if (i // batch_size) % 100 == 0 and i > 0:
            print(f"  - 已处理 {i}/{n_samples}...")

    log(f"分配完成，耗时 {time.time()-t1:.2f}s")
    
    # --- 阶段3：统计与写入 ---
    cluster_points = {i: [] for i in range(K)}
    for idx, label in enumerate(final_labels):
        cluster_points[label].append(idx)
    
    cluster_sizes = [len(cluster_points[i]) for i in range(K)]
    log(f"每个聚类的大小: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.2f}")
    
    # 计算平均距离 (L2 distance squared sum / n_samples)
    avg_distance = total_dist_sq / n_samples
    log(f"平均点到聚类中心的距离 (L2 sq): {avg_distance:.4f}")

    log(f"保存聚类结果到: {centroids_file}")
    write_centroids(centroids_file, K, final_labels, centroids, cluster_points)
    log("普通 K-means 聚类完成！")

if __name__ == "__main__":
    main()