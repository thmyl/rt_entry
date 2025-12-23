import faiss
import numpy as np

d = 128
n_db = 1000000
db_vectors = np.random.rand(n_db, d).astype(np.float32)
query_vectors = np.random.rand(100, d).astype(np.float32)

# 1. 创建基础近似索引（IVF-PQ）
coarse_quantizer = faiss.IndexFlatL2(d)
base_index = faiss.IndexIVFPQ(coarse_quantizer, d, 4096, 16, 8)
# base_index.train(db_vectors)
# base_index.add(db_vectors)
base_index.nprobe = 20

# 2. 创建精修索引（IndexRefineFlat）
refine_index = faiss.IndexRefineFlat(base_index)
refine_index.k_factor = 10 
# 精修索引需存储原始向量（用于精确计算）
refine_index.train(db_vectors)
refine_index.add(db_vectors)

print("base_index.ntotal:", base_index.ntotal)   # 输出：1000000
print("refine_index.ntotal:", refine_index.ntotal) # 输出：1000000

# 3. 检索（自动先近似检索，再精修）
k = 10
# 基础索引检索：返回K*10=100个候选结果
D_base, I_base = base_index.search(query_vectors, k*10)
# 精修索引检索：对100个候选结果精确计算，返回Top-10
D_refine, I_refine = refine_index.search(query_vectors, k)

# 4. 对比精度（与IndexFlat对比）
benchmark_index = faiss.IndexFlatL2(d)
benchmark_index.add(db_vectors)
D_bench, I_bench = benchmark_index.search(query_vectors, k)

# 计算基础索引与精修索引的召回率
def calculate_recall(I_pred, I_bench, k):
    recall = 0.0
    for i in range(len(I_pred)):
        pred = set(I_pred[i])
        bench = set(I_bench[i])
        recall += len(pred & bench) / k
    return recall / len(I_pred)

recall_base = calculate_recall(I_base[:, :k], I_bench, k)
recall_refine = calculate_recall(I_refine, I_bench, k)

print(f"基础索引召回率：{recall_base:.4f}")  # 约0.92
print(f"精修索引召回率：{recall_refine:.4f}")  # 约0.98
