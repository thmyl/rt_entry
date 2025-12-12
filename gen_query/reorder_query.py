import numpy as np
import faiss
import struct

TOTAL_QUERIES = 10000
QUERY_PATH = "/data/myl/sift1M/sift1M_query.fvecs"
OUTPUT_PATH = "sift1M_query.fvecs"
KNN_K = 32

def log(msg):
    print(msg)


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

query_vectors = read_vectors(QUERY_PATH)
N, D = query_vectors.shape
print(f"Query vectors shape: {query_vectors.shape}")

index = faiss.IndexFlatL2(D)
index.add(query_vectors)
_, knn = index.search(query_vectors, KNN_K + 1) # 包含自身
knn = knn[:, 1:] # 去掉自身

print(f"Query graph constructed: {N} nodes, each with {KNN_K} neighbors.")

def random_walk_knn(knn, total_queries):
    """随机游走生成不重复的query IDs
    Args:
        knn: KNN图，shape为(n, k)，每行是某个节点的k个邻居
        total_queries: 需要生成的query数量
    Returns:
        walk_ids: 不重复的节点ID列表
    """
    walk_ids = []
    visited = set() # 记录已访问的节点
    cur = np.random.randint(0, N) # 起始节点ID
    cnt = 0
    random_cnt = 0
    for _ in range(total_queries):
        walk_ids.append(cur)
        cnt += 1
        visited.add(cur)
        unvisited_neighbors = [neighbor for neighbor in knn[cur] if neighbor not in visited]
        if len(unvisited_neighbors) > 0:
            cur = np.random.choice(unvisited_neighbors)
        else:
            # print(f"No unvisited neighbors for node {cur}, reselecting...")
            unvisited_nodes = [i for i in range(N) if i not in visited]
            if not unvisited_nodes:
                # 已经走遍所有节点，提前结束
                print(f"Reached all nodes, total count: {cnt}")
                break
            cur = np.random.choice(unvisited_nodes)
            random_cnt += 1
    print(f"Randomly selected nodes: {random_cnt}")
    return walk_ids

walk_ids = random_walk_knn(knn, TOTAL_QUERIES)
print(f"Generated {len(walk_ids)} unique query IDs.")

# 保存walk_ids
query_indices = query_vectors[walk_ids]
write_fvecs(OUTPUT_PATH, query_indices)