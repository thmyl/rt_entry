#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
普通 K-means 聚类程序：使用 faiss 对 dataset 进行聚类（不做平衡分配）
"""

import sys
import numpy as np
import faiss
import struct
import os
import time

def read_fvecs(filename):
    """读取fvecs格式文件"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vector_bytes = f.read(dim * 4)
            if not vector_bytes:
                break
            vector = struct.unpack('f' * dim, vector_bytes)
            vectors.append(np.array(vector, dtype=np.float32))
    return np.array(vectors)

def read_fbin(filename):
    """读取fbin格式文件"""
    with open(filename, 'rb') as f:
        # 读取向量数量 n (4字节 int)
        n_bytes = f.read(4)
        if not n_bytes:
            raise ValueError("无法读取文件头: n")
        n = struct.unpack('i', n_bytes)[0]
        
        # 读取维度 d (4字节 int)
        d_bytes = f.read(4)
        if not d_bytes:
            raise ValueError("无法读取文件头: d")
        d = struct.unpack('i', d_bytes)[0]
        
        # 读取所有向量数据 (n*d 个 float，每个4字节)
        data_bytes = f.read(n * d * 4)
        if len(data_bytes) != n * d * 4:
            raise ValueError(f"数据不完整: 期望 {n * d * 4} 字节，实际读取 {len(data_bytes)} 字节")
        
        # 将二进制数据转换为 numpy 数组
        vectors = np.frombuffer(data_bytes, dtype=np.float32).reshape(n, d)
    
    return vectors

def write_centroids(centroids_file, n_clusters, labels, centroids, cluster_points):
    """写入聚类结果到文件（保持原有二进制格式不变）"""
    with open(centroids_file, 'wb') as f:
        # 写入聚类个数
        f.write(struct.pack('i', n_clusters))
        # 写入每个点的聚类标签
        for label in labels:
            f.write(struct.pack('i', label))
        # 写入聚类中心
        for centroid in centroids:
            for val in centroid:
                f.write(struct.pack('f', val))
        # 写入每个聚类包含的点的下标
        for cluster_id in range(n_clusters):
            point_ids = cluster_points[cluster_id]
            f.write(struct.pack('i', len(point_ids)))
            for point_id in point_ids:
                f.write(struct.pack('i', point_id))

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/cluster.py <K>")
        print("  K: 聚类个数，默认为10")
        sys.exit(1)
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    # dataset_path = "/data/myl/sift1M/sift1M_base.fvecs"  # TODO: change dataset path
    # dataset_path = "/data/myl/deep1M/deep1M_base.fvecs" # TODO: change dataset path
    # dataset_path = "/data/myl/sift100M/sift100M_base.fbin"  # TODO: change dataset path
    dataset_path = "/data/myl/deep100M/fbin/deep100M_base.fbin"  # TODO: change dataset path
    dataset_name = os.path.basename(dataset_path).split('_')[0]
    data_root = os.path.join("/data/myl/cache_search/data", dataset_name)
    # os.makedirs("data", exist_ok=True)
    os.makedirs(data_root, exist_ok=True)
    centroids_file = os.path.join(data_root, f"centroids_{K}")
    # 保持输出 centroids 文件名格式不变，仅修改日志文件名
    log_file = os.path.join(data_root, "cluster.log")
    def log(msg):
        print(msg)
        with open(log_file, 'a') as lf:
            lf.write(str(msg) + "\n")

    if os.path.exists(centroids_file):
        log(f"centroids_{K}文件已存在，跳过聚类计算: {centroids_file}")
        return

    log(f"开始读取数据集: {dataset_path}")
    # 根据文件后缀选择读取函数
    if dataset_path.endswith('.fbin'):
        dataset = read_fbin(dataset_path)
    elif dataset_path.endswith('.fvecs'):
        dataset = read_fvecs(dataset_path)
    else:
        # 默认尝试使用 fvecs 格式
        log(f"警告: 未知文件格式，尝试使用 fvecs 格式读取")
        dataset = read_fvecs(dataset_path)
    n_samples = dataset.shape[0]
    d = dataset.shape[1]
    log(f"数据集大小: {dataset.shape}")

    log(f"开始普通 K-means 聚类，K={K}")

    # 第一步：使用 faiss 进行 K-means 聚类
    log(f"第1步: 使用 faiss 训练 K-means 聚类")
    kmeans = faiss.Kmeans(d, K, niter=20, verbose=True, gpu=False)
    kmeans.train(dataset)
    centroids = kmeans.centroids
    log(f"K-means 训练完成，获得 {K} 个聚类中心")

    # 第二步：使用 faiss 的 index 对所有样本做最近中心分配（普通最近中心分配）
    log(f"第2步: 计算每个点所属的最近聚类中心")
    # kmeans.index 在 train 后已经包含聚类中心
    distances, assign = kmeans.index.search(dataset, 1)  # distances: [n_samples,1]
    labels = assign.reshape(-1).astype(np.int32)
    log(f"分配完成，标签范围: {labels.min()} - {labels.max()}")
    
    # 统计每个聚类的点数
    cluster_points = {i: [] for i in range(K)}
    for idx, label in enumerate(labels):
        cluster_points[label].append(idx)
    
    # 统计每个聚类的点数（不再强制平衡）
    cluster_sizes = [len(cluster_points[i]) for i in range(K)]
    log(f"每个聚类的大小: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.2f}")
    log(f"聚类大小分布: {cluster_sizes[:10]}..." if K > 10 else f"聚类大小分布: {cluster_sizes}")
    
    # 计算总体聚类质量（平均距离）
    avg_distance = float(distances.mean())
    log(f"平均点到聚类中心的距离: {avg_distance:.4f}")

    log(f"保存聚类结果到: {centroids_file}")
    write_centroids(centroids_file, K, labels, centroids, cluster_points)
    log("普通 K-means 聚类完成！")

if __name__ == "__main__":
    main()

