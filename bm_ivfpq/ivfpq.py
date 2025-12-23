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
k = 2048
nprobe = 2000
gpu_id = 1
index_path = f"ivfpq_sift1M_{nlist}_M{M}_nbits{nbits}.index"

# -------------------------
# 构建 CPU IVF-PQ
# -------------------------
print("Building CPU IVF-PQ index...")
quantizer = faiss.IndexFlatL2(d)
index_cpu = faiss.IndexIVFPQ(
    quantizer,
    d,
    nlist,
    M,
    nbits
)

index_cpu.nprobe = nprobe

# -------------------------
# 训练索引 + 保存索引
# -------------------------
if not os.path.exists(index_path):
    print("Index file not found. Training CPU IVF-PQ index...")
    index_cpu.train(xb)
    index_cpu.add(xb)
    faiss.write_index(index_cpu, index_path)
    print(f"Index trained and saved to {index_path}.")
else:
    print(f"Index file {index_path} exists. Loading index...")
    index_cpu = faiss.read_index(index_path)

# -------------------------
# 转移到 GPU
# -------------------------
print("Moving index to GPU...")
res = faiss.StandardGpuResources()
index_gpu = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)

# ⚠️ 再次确认 nprobe（GPU index 有时需要重新设）
index_gpu.nprobe = nprobe

# -------------------------
# 搜索（不使用 re-ranking）
# -------------------------
print("Searching...")
D, I = index_gpu.search(xq, k)   # I: (nq, 128)

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
