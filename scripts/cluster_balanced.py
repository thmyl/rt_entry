#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平衡聚类程序：使用faiss对dataset进行初始聚类，然后重新分配以实现平衡
每个聚类包含 n/k 个点
"""

import sys
import numpy as np
import faiss
import struct
import os
import time
from scipy.optimize import linear_sum_assignment

# 尝试导入 GPU 加速库
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: torch 未安装，无法使用 GPU 加速")

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

def write_centroids(centroids_file, n_clusters, labels, centroids, cluster_points):
    """写入聚类结果到文件"""
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

def balanced_assignment(distances, cluster_size):
    """
    使用贪心策略进行平衡分配
    distances: [n_samples, n_clusters] 距离矩阵
    cluster_size: 每个聚类应包含的点数
    返回: labels [n_samples]
    """
    n_samples, n_clusters = distances.shape
    labels = np.full(n_samples, -1, dtype=np.int32)
    cluster_counts = np.zeros(n_clusters, dtype=np.int32)
    
    # 为每个点找到距离排序
    sorted_clusters = np.argsort(distances, axis=1)
    
    # 贪心分配：优先处理选择较少的点
    # 计算每个点的"竞争度"（有多少个可行的聚类选择）
    assigned = np.zeros(n_samples, dtype=bool)
    
    for iteration in range(n_samples):
        # 找到未分配且竞争度最小的点
        best_point = -1
        best_priority = float('inf')
        
        for i in range(n_samples):
            if assigned[i]:
                continue
            
            # 找到第一个还有空位的聚类
            available_rank = 0
            for rank in range(n_clusters):
                cluster = sorted_clusters[i, rank]
                if cluster_counts[cluster] < cluster_size:
                    available_rank = rank
                    break
            
            if available_rank < best_priority:
                best_priority = available_rank
                best_point = i
        
        if best_point == -1:
            break
        
        # 分配 best_point 到最近的还有空位的聚类
        for rank in range(n_clusters):
            cluster = sorted_clusters[best_point, rank]
            if cluster_counts[cluster] < cluster_size:
                labels[best_point] = cluster
                cluster_counts[cluster] += 1
                assigned[best_point] = True
                break
    
    return labels

def balanced_assignment_batch(distances, cluster_size):
    """
    使用批量匹配进行平衡分配（更快但可能略微次优）
    """
    n_samples, n_clusters = distances.shape
    labels = np.full(n_samples, -1, dtype=np.int32)
    
    # 对于每个聚类，选择距离最近的 cluster_size 个点
    assigned = np.zeros(n_samples, dtype=bool)
    
    for cluster_id in range(n_clusters):
        # 获取到该聚类的距离
        cluster_distances = distances[:, cluster_id].copy()
        cluster_distances[assigned] = np.inf  # 已分配的点设为无穷大
        
        # 选择最近的 cluster_size 个点
        nearest_points = np.argpartition(cluster_distances, min(cluster_size, n_samples - assigned.sum()))[:cluster_size]
        
        for point_id in nearest_points:
            if not assigned[point_id]:
                labels[point_id] = cluster_id
                assigned[point_id] = True
    
    # 处理可能未分配的点（边界情况）
    unassigned = np.where(~assigned)[0]
    for point_id in unassigned:
        # 找到最近的有空间的聚类（理论上不应该发生）
        cluster_id = np.argmin(distances[point_id])
        labels[point_id] = cluster_id
    
    return labels

def balanced_assignment_fast(distances, cluster_size):
    """
    快速批量平衡分配 - O(n*k) 复杂度，比原版快很多
    改进版本：避免不必要的复制和循环
    """
    n_samples, n_clusters = distances.shape
    labels = np.full(n_samples, -1, dtype=np.int32)
    assigned = np.zeros(n_samples, dtype=bool)
    
    for cluster_id in range(n_clusters):
        # 获取未分配点到该聚类的距离
        cluster_distances = distances[:, cluster_id].copy()
        cluster_distances[assigned] = np.inf
        
        # 快速找到最近的 cluster_size 个点
        n_to_select = min(cluster_size, n_samples - assigned.sum())
        if n_to_select <= 0:
            break
            
        # argpartition 比完全排序快得多 - O(n) vs O(n log n)
        indices = np.argpartition(cluster_distances, n_to_select - 1)[:n_to_select]
        # 只选择真正未分配的点
        valid_indices = indices[~assigned[indices]][:n_to_select]
        
        labels[valid_indices] = cluster_id
        assigned[valid_indices] = True
    
    return labels

def balanced_assignment_gpu(distances, cluster_size):
    """
    使用 PyTorch GPU 加速的平衡分配
    适用于大规模数据集 (百万级样本)
    """
    if not TORCH_AVAILABLE:
        print("PyTorch 不可用，回退到 CPU 版本")
        return balanced_assignment_fast(distances, cluster_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA 不可用，使用 CPU 版本")
        return balanced_assignment_fast(distances, cluster_size)
    
    n_samples, n_clusters = distances.shape
    
    # 将数据转移到 GPU
    distances_gpu = torch.from_numpy(distances).to(device)
    labels = torch.full((n_samples,), -1, dtype=torch.int32, device=device)
    assigned = torch.zeros(n_samples, dtype=torch.bool, device=device)
    
    for cluster_id in range(n_clusters):
        # GPU 上的距离计算
        cluster_distances = distances_gpu[:, cluster_id].clone()
        cluster_distances[assigned] = float('inf')
        
        # GPU 上快速找到最小值
        n_to_select = min(cluster_size, n_samples - assigned.sum().item())
        if n_to_select <= 0:
            break
        
        # topk 在 GPU 上非常快
        _, indices = torch.topk(cluster_distances, n_to_select, largest=False)
        
        # 过滤已分配的点
        mask = ~assigned[indices]
        valid_indices = indices[mask][:n_to_select]
        
        labels[valid_indices] = cluster_id
        assigned[valid_indices] = True
    
    # 转回 CPU 和 numpy
    return labels.cpu().numpy()

def balanced_assignment_progressive(distances, cluster_size, n_rounds=3):
    """
    渐进式平衡分配 - 多轮迭代优化
    第一轮快速分配，后续轮次调整以提高质量
    """
    n_samples, n_clusters = distances.shape
    
    # 第一轮：快速批量分配
    labels = balanced_assignment_fast(distances, cluster_size)
    
    # 后续轮次：局部优化
    for round_idx in range(1, n_rounds):
        improved = False
        
        # 对每个点，检查是否有更近的聚类（且该聚类有空位）
        cluster_counts = np.bincount(labels, minlength=n_clusters)
        
        for point_id in np.random.permutation(n_samples):
            current_cluster = labels[point_id]
            current_distance = distances[point_id, current_cluster]
            
            # 找到更近的聚类
            sorted_clusters = np.argsort(distances[point_id])
            for new_cluster in sorted_clusters:
                if new_cluster == current_cluster:
                    break
                    
                # 检查新聚类是否有空位或可以交换
                if cluster_counts[new_cluster] < cluster_size:
                    # 直接移动
                    labels[point_id] = new_cluster
                    cluster_counts[current_cluster] -= 1
                    cluster_counts[new_cluster] += 1
                    improved = True
                    break
                elif distances[point_id, new_cluster] < current_distance * 0.9:
                    # 尝试找一个可以移出的点
                    candidates = np.where(labels == new_cluster)[0]
                    for candidate in candidates:
                        if (distances[candidate, current_cluster] < distances[candidate, new_cluster] and
                            cluster_counts[current_cluster] < cluster_size):
                            # 交换
                            labels[point_id] = new_cluster
                            labels[candidate] = current_cluster
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            break
    
    return labels

def balanced_assignment_with_remainder(distances, cluster_size, remainder, use_gpu=False):
    """
    处理有余数情况的平衡分配
    前 remainder 个聚类分配 cluster_size+1 个点，其余聚类分配 cluster_size 个点
    支持 GPU 加速
    """
    n_samples, n_clusters = distances.shape
    
    # 检查是否使用 GPU
    if use_gpu and TORCH_AVAILABLE:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            return _balanced_assignment_with_remainder_gpu(distances, cluster_size, remainder, device)
    
    # CPU 版本
    return _balanced_assignment_with_remainder_cpu(distances, cluster_size, remainder)

def _balanced_assignment_with_remainder_cpu(distances, cluster_size, remainder):
    """CPU 版本的有余数平衡分配"""
    n_samples, n_clusters = distances.shape
    labels = np.full(n_samples, -1, dtype=np.int32)
    assigned = np.zeros(n_samples, dtype=bool)
    
    # 为每个聚类确定应分配的点数
    cluster_sizes = np.full(n_clusters, cluster_size, dtype=np.int32)
    cluster_sizes[:remainder] = cluster_size + 1
    
    # 按聚类分配
    for cluster_id in range(n_clusters):
        cluster_distances = distances[:, cluster_id].copy()
        cluster_distances[assigned] = np.inf
        
        target_size = cluster_sizes[cluster_id]
        n_to_select = min(target_size, n_samples - assigned.sum())
        if n_to_select <= 0:
            break
        
        # 使用 argpartition 快速选择
        indices = np.argpartition(cluster_distances, n_to_select - 1)[:n_to_select]
        valid_indices = indices[~assigned[indices]][:n_to_select]
        
        labels[valid_indices] = cluster_id
        assigned[valid_indices] = True
    
    # 处理未分配的点
    unassigned = np.where(~assigned)[0]
    if len(unassigned) > 0:
        sorted_clusters = np.argsort(distances[unassigned], axis=1)
        cluster_counts = np.bincount(labels[labels >= 0], minlength=n_clusters)
        
        for point_id in unassigned:
            for cluster in sorted_clusters[point_id - unassigned[0]]:
                if cluster_counts[cluster] < cluster_sizes[cluster]:
                    labels[point_id] = cluster
                    cluster_counts[cluster] += 1
                    break
    
    return labels

def _balanced_assignment_with_remainder_gpu(distances, cluster_size, remainder, device):
    """GPU 版本的有余数平衡分配"""
    n_samples, n_clusters = distances.shape
    
    # 将数据转移到 GPU
    distances_gpu = torch.from_numpy(distances).to(device)
    labels = torch.full((n_samples,), -1, dtype=torch.int32, device=device)
    assigned = torch.zeros(n_samples, dtype=torch.bool, device=device)
    
    # 为每个聚类确定应分配的点数
    cluster_sizes = torch.full((n_clusters,), cluster_size, dtype=torch.int32, device=device)
    cluster_sizes[:remainder] = cluster_size + 1
    
    # 按聚类分配
    for cluster_id in range(n_clusters):
        # GPU 上的距离计算
        cluster_distances = distances_gpu[:, cluster_id].clone()
        cluster_distances[assigned] = float('inf')
        
        target_size = cluster_sizes[cluster_id].item()
        n_to_select = min(target_size, n_samples - assigned.sum().item())
        if n_to_select <= 0:
            break
        
        # GPU 上使用 topk 快速找到最小的 n_to_select 个
        _, indices = torch.topk(cluster_distances, n_to_select, largest=False)
        
        # 过滤已分配的点
        mask = ~assigned[indices]
        valid_indices = indices[mask][:n_to_select]
        
        if len(valid_indices) > 0:
            labels[valid_indices] = cluster_id
            assigned[valid_indices] = True
    
    # 处理未分配的点
    unassigned_mask = ~assigned
    n_unassigned = unassigned_mask.sum().item()
    
    if n_unassigned > 0:
        unassigned_indices = torch.where(unassigned_mask)[0]
        unassigned_distances = distances_gpu[unassigned_indices]
        
        # 对每个未分配点的距离排序
        sorted_clusters = torch.argsort(unassigned_distances, dim=1)
        
        # 统计每个聚类的当前点数
        cluster_counts = torch.zeros(n_clusters, dtype=torch.int32, device=device)
        for c in range(n_clusters):
            cluster_counts[c] = (labels == c).sum()
        
        # 为每个未分配点找到合适的聚类
        for i, point_id in enumerate(unassigned_indices):
            for cluster in sorted_clusters[i]:
                cluster_idx = cluster.item()
                if cluster_counts[cluster_idx] < cluster_sizes[cluster_idx]:
                    labels[point_id] = cluster_idx
                    cluster_counts[cluster_idx] += 1
                    break
    
    # 转回 CPU 和 numpy
    return labels.cpu().numpy()

def refine_centroids(dataset, labels, n_clusters):
    """根据新的分配重新计算聚类中心"""
    centroids = np.zeros((n_clusters, dataset.shape[1]), dtype=np.float32)
    for i in range(n_clusters):
        cluster_points = dataset[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = cluster_points.mean(axis=0)
    return centroids

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/cluster_balanced.py <K>")
        print("  K: 聚类个数，默认为10")
        sys.exit(1)
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    dataset_path = "/data/myl/sift1M/sift1M_base.fvecs" # TODO: change dataset path
    # dataset_path = "/data/myl/deep1M/deep1M_base.fvecs" # TODO: change dataset path
    dataset_name = os.path.basename(dataset_path).split('_')[0]
    data_root = os.path.join("data", dataset_name)
    os.makedirs("data", exist_ok=True)
    os.makedirs(data_root, exist_ok=True)
    centroids_file = os.path.join(data_root, f"centroids_{K}")
    log_file = os.path.join(data_root, "cluster_balanced.log")
    def log(msg):
        print(msg)
        with open(log_file, 'a') as lf:
            lf.write(str(msg) + "\n")

    if os.path.exists(centroids_file):
        log(f"centroids_{K}文件已存在，跳过聚类计算: {centroids_file}")
        return

    log(f"开始读取数据集: {dataset_path}")
    dataset = read_fvecs(dataset_path)
    n_samples = dataset.shape[0]
    d = dataset.shape[1]
    log(f"数据集大小: {dataset.shape}")

    # 计算每个聚类应该包含的点数
    cluster_size = n_samples // K
    remainder = n_samples % K
    log(f"开始平衡K-means聚类，K={K}, 每个聚类目标大小={cluster_size}, 余数={remainder}")

    # 第一步：使用faiss进行初始聚类，获得聚类中心
    log(f"第1步: 使用faiss进行初始K-means聚类")
    kmeans = faiss.Kmeans(d, K, niter=20, verbose=True, gpu=False)
    kmeans.train(dataset)
    initial_centroids = kmeans.centroids
    log(f"初始聚类完成，获得{K}个聚类中心")

    # 第二步：计算所有点到所有聚类中心的距离
    log(f"第2步: 计算距离矩阵")
    index = faiss.IndexFlatL2(d)
    index.add(initial_centroids)
    distances, _ = index.search(dataset, K)  # [n_samples, K]
    log(f"距离计算完成")

    # 第三步：使用快速算法重新分配点，确保平衡
    log(f"第3步: 使用快速平衡分配算法")
    
    start_time = time.time()
    
    # 选择分配算法（按优先级）
    # 详细检查 GPU 可用性
    log(f"  GPU 可用性检查:")
    log(f"    - PyTorch 是否已安装: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        cuda_available = torch.cuda.is_available()
        log(f"    - CUDA 是否可用: {cuda_available}")
        if cuda_available:
            log(f"    - GPU 设备名称: {torch.cuda.get_device_name(0)}")
            log(f"    - GPU 设备数量: {torch.cuda.device_count()}")
    else:
        log(f"    - CUDA 检查: 跳过 (PyTorch 未安装)")
    
    use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
    log(f"    - 最终决策: {'使用 GPU' if use_gpu else '使用 CPU'}")
    
    if use_gpu and n_samples > 100000:  # 大规模数据集使用GPU
        log(f"  使用 GPU 加速算法 (数据量: {n_samples})")
        if remainder == 0:
            labels = balanced_assignment_gpu(distances, cluster_size)
        else:
            labels = balanced_assignment_with_remainder(distances, cluster_size, remainder, use_gpu=True)
    else:
        # CPU 快速算法
        if not use_gpu:
            if not TORCH_AVAILABLE:
                log(f"  使用 CPU 快速算法 - 原因: PyTorch 未安装")
            elif not torch.cuda.is_available():
                log(f"  使用 CPU 快速算法 - 原因: CUDA 不可用")
        elif n_samples <= 100000:
            log(f"  使用 CPU 快速算法 - 原因: 数据量较小 ({n_samples} <= 100000)")
        else:
            log(f"  使用 CPU 快速算法 (数据量: {n_samples})")
            
        if remainder == 0:
            labels = balanced_assignment_fast(distances, cluster_size)
        else:
            labels = balanced_assignment_with_remainder(distances, cluster_size, remainder, use_gpu=False)
    
    elapsed_time = time.time() - start_time
    log(f"平衡分配完成，耗时: {elapsed_time:.2f}秒，标签范围: {labels.min()} - {labels.max()}")

    # 第四步：根据新分配重新计算聚类中心
    log(f"第4步: 根据新分配重新计算聚类中心")
    centroids = refine_centroids(dataset, labels, K)
    
    # 统计每个聚类的点数
    cluster_points = {i: [] for i in range(K)}
    for idx, label in enumerate(labels):
        cluster_points[label].append(idx)
    
    # 验证平衡性
    cluster_sizes = [len(cluster_points[i]) for i in range(K)]
    log(f"每个聚类的大小: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.2f}")
    log(f"聚类大小分布: {cluster_sizes[:10]}..." if K > 10 else f"聚类大小分布: {cluster_sizes}")
    
    # 计算总体聚类质量（平均距离）
    total_distance = 0
    for i in range(n_samples):
        total_distance += np.sum((dataset[i] - centroids[labels[i]]) ** 2)
    avg_distance = total_distance / n_samples
    log(f"平均点到聚类中心的距离: {avg_distance:.4f}")

    log(f"保存聚类结果到: {centroids_file}")
    write_centroids(centroids_file, K, labels, centroids, cluster_points)
    log("平衡聚类完成！")

if __name__ == "__main__":
    main()

