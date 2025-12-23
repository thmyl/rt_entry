import faiss
import numpy as np
import os

# -------------------------
# 路径配置
# -------------------------
base_path = "/data/myl/sift1M/sift1M_base.fvecs"
query_path = "/data/myl/sift1M/sift1M_query.fvecs"
gt_path = "/data/myl/sift1M/sift1M_groundtruth.ivecs"

# -------------------------
# 工具函数：读取 fvecs / ivecs
# -------------------------
def read_fvecs(fname):
    with open(fname, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
    dim = data[0]
    data = data.reshape(-1, dim + 1)
    return data[:, 1:].astype(np.float32)

def read_ivecs(fname):
    with open(fname, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
    dim = data[0]
    data = data.reshape(-1, dim + 1)
    return data[:, 1:]

# -------------------------
# 加载数据
# -------------------------
print("Loading data...")
xb = read_fvecs(base_path)     # (1e6, 128)
xq = read_fvecs(query_path)    # (1e4, 128)
gt = read_ivecs(gt_path)       # (1e4, 100)

nb, d = xb.shape
nq = xq.shape[0]

print(f"Base: {xb.shape}, Query: {xq.shape}, GT: {gt.shape}")

# -------------------------
# IVF-PQ 参数
# -------------------------
nlist = 4096
M = 32
nbits = 8
k = 128
nprobe = 2000
gpu_id = 1
k_reorder = 2048       # 精排候选数，>= k
index_path = f"ivfpqr_sift1M_{nlist}_M{M}_nbits{nbits}.index"

# -------------------------
# 构建 CPU IVF-PQ
# -------------------------
print("Building CPU IVF-PQ index...")
quantizer = faiss.IndexFlatL2(d)
ivfpq = faiss.IndexIVFPQ(
    quantizer,
    d,
    nlist,
    M,
    nbits
)
ivfpq.nprobe = nprobe

# -------------------------
# 训练索引 + 保存索引
# -------------------------
if not os.path.exists(index_path):
    print("Index file not found. Training CPU IVF-PQ index...")
    ivfpq.train(xb)
    ivfpq.add(xb)
    refine_index = faiss.IndexRefineFlat(ivfpq, xb)
    # refine_index.train(xb)  # 训练精炼索引
    # refine_index.add(xb)  # 将数据添加到内部的精炼索引
    faiss.write_index(refine_index, index_path)
    print(f"Index trained and saved to {index_path}.")
else:
    print(f"Index file {index_path} exists. Loading index...")
    refine_index = faiss.read_index(index_path)
    
# -------------------------
# 包一层精排（Re-ranking）
# -------------------------
# refine_index = faiss.IndexRefineFlat(ivfpq)
# refine_index.train(xb)  # 训练精炼索引
# refine_index.add(xb)  # 将数据添加到内部的精炼索引
refine_index.k_factor = k_reorder // k  # k_reorder = k * k_factor
refine_index.nprobe = nprobe

# -------------------------
# 转移到 GPU
# -------------------------
# print("Moving index to GPU...")
# res = faiss.StandardGpuResources()
# index_gpu = faiss.index_cpu_to_gpu(res, gpu_id, refine_index)

# index_gpu.nprobe = nprobe

# -------------------------
# 搜索（使用 re-ranking）
# -------------------------
print("Searching with IVF-PQ + re-ranking...")
# D, I = index_gpu.search(xq, k)   # I: (nq, 128)
D, I = refine_index.search(xq, k)   # I: (nq, 128)

# -------------------------
# 计算 recall10@128
# -------------------------
print("Computing recall10@128...")
correct = 0
for i in range(nq):
    gt10 = set(gt[i, :10])
    res128 = set(I[i])
    correct += len(gt10 & res128)

recall10_128 = correct / (nq * 10)
print(f"Recall10@128 = {recall10_128:.6f}")
