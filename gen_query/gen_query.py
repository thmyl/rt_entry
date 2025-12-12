import numpy as np
import faiss
import struct

# ==========================
# 配置参数
# ==========================
TOTAL_QUERIES = 10000     # 总 query 数
SAMPLE_SIZE = 20000       # 用于构建 query graph 的采样点数量
KNN_K = 100                # KNN 图邻居数量
# DATASET_PATH = "/data/myl/sift1M/sift1M_base.fvecs"
DATASET_PATH = "/data/myl/sift1B/bigann_base.bvecs"
LOG_FILE = "/data/myl/cache_search/gen_query/gen_query.log"
def log(msg):
    print(msg)
    with open(LOG_FILE, 'a') as lf:
        lf.write(str(msg) + "\n")

# ==========================
# Step 0: base vectors
# ==========================
# base_vectors: numpy array, shape = (N, D)
# 这里假设你已经有 base_vectors，比如 SIFT1M
# base_vectors = np.load("sift_base.npy")

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

def write_fvecs(filename, vectors):
    """将向量保存为fvecs格式文件
    Args:
        filename: 输出文件名
        vectors: numpy数组，shape为(nq, dim)，dtype应为float32
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    nq, dim = vectors.shape
    
    with open(filename, 'wb') as f:
        for i in range(nq):
            # 先写入维度dim（4字节整数）
            f.write(struct.pack('i', dim))
            # 然后写入dim维的float向量（每个float32占4字节）
            f.write(vectors[i].tobytes())

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

def read_bvecs(filename):
    """读取bvecs格式文件"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vector_bytes = f.read(dim)
            if not vector_bytes:
                break
            vector = struct.unpack('B' * dim, vector_bytes)
            vectors.append(np.array(vector, dtype=np.uint8))
    return np.array(vectors)

def read_vectors(filename):
    """读取向量文件，统一转换为float32格式
    Args:
        filename: 文件路径
        use_memmap: 对于fbin格式，是否使用内存映射（适合大文件）
    """
    if filename.endswith('.fbin'):
        vectors = read_fbin(filename)
    elif filename.endswith('.fvecs'):
        vectors = read_fvecs(filename)
    elif filename.endswith('.bvecs'):
        vectors = read_bvecs(filename)  # 读取为uint8
        # 转换为float32
        vectors = vectors.astype(np.float32)
    else:
        raise ValueError(f"未知文件格式: {filename}")
    
    return vectors.astype(np.float32)

# 根据文件后缀选择读取函数
# if DATASET_PATH.endswith('.fbin'):
#     base_vectors = read_fbin(DATASET_PATH)
# elif DATASET_PATH.endswith('.fvecs'):
#     base_vectors = read_fvecs(DATASET_PATH)
# elif DATASET_PATH.endswith('.bvecs'):
#     base_vectors = read_bvecs(DATASET_PATH)
# else:
#     # 默认尝试使用 fvecs 格式
#     log(f"警告: 未知文件格式，尝试使用 fvecs 格式读取")
#     base_vectors = read_fvecs(DATASET_PATH)
base_vectors = read_vectors(DATASET_PATH)

N, D = base_vectors.shape
print(f"base_vectors shape: {base_vectors.shape}")

# ==========================
# Step 1: 随机采样子集，构建 query graph
# ==========================
sample_indices = np.random.choice(N, SAMPLE_SIZE, replace=False)
sample_vectors = base_vectors[sample_indices]

print(f"sample_vectors shape: {sample_vectors.shape}")

# 使用 Faiss 构建 KNN 图
index = faiss.IndexFlatL2(D)
index.add(sample_vectors)
_, knn = index.search(sample_vectors, KNN_K + 1)  # 包含自身
knn = knn[:, 1:]  # 去掉自身

print(f"Query graph constructed: {SAMPLE_SIZE} nodes, each with {KNN_K} neighbors.")

# # ==========================
# # Step 2: Random walk 生成 query IDs
# # ==========================
# def random_walk_knn(knn, start_id, total_queries):
#     """随机游走生成不重复的query IDs
    
#     Args:
#         knn: KNN图，shape为(n, k)，每行是某个节点的k个邻居
#         start_id: 起始节点ID
#         total_queries: 需要生成的query数量
    
#     Returns:
#         walk_ids: 不重复的节点ID列表
#     """
#     walk_ids = []
#     visited = set()  # 记录已访问的节点
#     cur = start_id
#     n_nodes = len(knn)  # 总节点数
#     cnt = 0 # 记录重新随机选择的节点数
#     for _ in range(total_queries):
#         walk_ids.append(cur)
#         visited.add(cur)
        
#         # 如果已经访问了所有节点，无法继续生成不重复的ID
#         if len(visited) >= n_nodes:
#             raise ValueError(f"无法生成 {total_queries} 个不重复的ID，总节点数只有 {n_nodes}")
        
#         # 获取当前节点的未访问邻居
#         unvisited_neighbors = [neighbor for neighbor in knn[cur] if neighbor not in visited]
        
#         if len(unvisited_neighbors) > 0:
#             # 如果有未访问的邻居，随机选择一个
#             cur = np.random.choice(unvisited_neighbors)
#         else:
#             # 如果所有邻居都已访问，从所有未访问节点中随机选择一个
#             print(f"所有邻居都已访问，从所有未访问节点中随机选择一个")
#             cnt += 1
#             unvisited_nodes = [i for i in range(n_nodes) if i not in visited]
#             cur = np.random.choice(unvisited_nodes)
#     print(f"重新随机选择的节点数: {cnt}")
#     return walk_ids

# ==========================
# Step 2: 改进版 Random walk 生成 query IDs
# ==========================
def improved_random_walk_knn(knn, total_queries, restart_prob=0.2, walk_length=50, degree_bias=True):
    """
    改进的随机游走生成不重复的 query IDs，避免总在高密度区域
    
    Args:
        knn: KNN图，shape=(n, k)
        total_queries: 需要生成的 query 数量
        restart_prob: 每步随机重启的概率
        walk_length: 每次从起点走几步
        degree_bias: 是否使用度反向采样
        
    Returns:
        walk_ids: 不重复的 query 节点 ID 列表
    """
    n_nodes = len(knn)
    walk_ids = []
    visited = set()  # 记录已访问的节点
    
    # 如果需要的query数量超过总节点数，无法生成不重复的ID
    if total_queries > n_nodes:
        raise ValueError(f"无法生成 {total_queries} 个不重复的ID，总节点数只有 {n_nodes}")
    
    while len(walk_ids) < total_queries:
        # 如果所有节点都已访问，无法继续
        if len(visited) >= n_nodes:
            raise ValueError(f"无法生成 {total_queries} 个不重复的ID，总节点数只有 {n_nodes}")
        
        # 随机选择未访问的起点
        unvisited_nodes = [i for i in range(n_nodes) if i not in visited]
        cur = np.random.choice(unvisited_nodes)
        
        # 每次短步游走
        for _ in range(walk_length):
            if cur not in visited:
                walk_ids.append(cur)
                visited.add(cur)
                if len(walk_ids) >= total_queries:
                    break
            
            # restart - 从未访问节点中随机选择
            if np.random.rand() < restart_prob:
                unvisited_nodes = [i for i in range(n_nodes) if i not in visited]
                if len(unvisited_nodes) == 0:
                    break
                cur = np.random.choice(unvisited_nodes)
                continue
            
            neighbors = knn[cur]
            if len(neighbors) == 0:
                unvisited_nodes = [i for i in range(n_nodes) if i not in visited]
                if len(unvisited_nodes) == 0:
                    break
                cur = np.random.choice(unvisited_nodes)
                continue
            
            # 获取未访问的邻居
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if len(unvisited_neighbors) == 0:
                # 如果所有邻居都已访问，从所有未访问节点中随机选择
                unvisited_nodes = [i for i in range(n_nodes) if i not in visited]
                if len(unvisited_nodes) == 0:
                    break
                cur = np.random.choice(unvisited_nodes)
                continue
            
            # degree biased - 只在未访问的邻居中选择
            if degree_bias:
                degrees = np.array([len(knn[n]) for n in unvisited_neighbors], dtype=float)
                probs = 1.0 / (degrees + 1e-6)  # 防止除零
                probs /= probs.sum()
                cur = np.random.choice(unvisited_neighbors, p=probs)
            else:
                cur = np.random.choice(unvisited_neighbors)
    
    return walk_ids[:total_queries]

# walk_ids = random_walk_knn(knn, start_node, TOTAL_QUERIES)
walk_ids = improved_random_walk_knn(knn, TOTAL_QUERIES)

# 验证生成的ID不重复
assert len(walk_ids) == len(set(walk_ids)), f"生成的ID有重复！总数量: {len(walk_ids)}, 唯一数量: {len(set(walk_ids))}"
print(f"成功生成 {len(walk_ids)} 个不重复的query IDs")

# 转换为原始 base vector 索引
query_indices = sample_indices[walk_ids]
query_vectors = base_vectors[query_indices]

# ==========================
# 保存到磁盘
# ==========================
# write_fvecs("sift1M_query.fvecs", query_vectors)
write_fvecs("sift1M_query.fvecs", sample_vectors)
# np.save("query_batches.npy", query_batches)
