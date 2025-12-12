import numpy as np
import faiss
import struct

# ==========================
# 配置参数
# ==========================
K = 100  # 每个query的前k个最近邻
QUERY_PATH = "sift1M_query.fvecs"
BASE_PATH = "/data/myl/sift1M/sift1M_base.fvecs"
OUTPUT_PATH = "sift1M_groundtruth.ivecs"

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

def write_ivecs(filename, neighbors):
    """将最近邻结果保存为ivecs格式文件
    
    Args:
        filename: 输出文件名
        neighbors: numpy数组，shape为(nq, k)，dtype应为int32，每行是k个邻居的id
    """
    neighbors = np.asarray(neighbors, dtype=np.int32)
    nq, k = neighbors.shape
    
    with open(filename, 'wb') as f:
        for i in range(nq):
            # 先写入k（4字节整数）
            f.write(struct.pack('i', k))
            # 然后写入k个邻居id（每个int32占4字节）
            f.write(neighbors[i].tobytes())

# ==========================
# 读取数据
# ==========================
print("正在读取query向量...")
query_vectors = read_fvecs(QUERY_PATH)
nq, d = query_vectors.shape
print(f"Query vectors shape: {query_vectors.shape}")

print("正在读取base向量...")
base_vectors = read_fvecs(BASE_PATH)
nb, db = base_vectors.shape
print(f"Base vectors shape: {base_vectors.shape}")

assert d == db, f"Query维度({d})与Base维度({db})不匹配！"

# ==========================
# 使用Faiss搜索最近邻
# ==========================
print(f"正在构建Faiss索引并搜索前{K}个最近邻...")
index = faiss.IndexFlatL2(d)
index.add(base_vectors.astype('float32'))

# 搜索每个query的前k个最近邻
distances, indices = index.search(query_vectors.astype('float32'), K)

print(f"搜索完成！")
print(f"结果形状: distances={distances.shape}, indices={indices.shape}")

# ==========================
# 保存ground truth
# ==========================
print(f"正在保存ground truth到 {OUTPUT_PATH}...")
write_ivecs(OUTPUT_PATH, indices)
print(f"Ground truth已保存！共 {nq} 个query，每个query有 {K} 个最近邻")

