import numpy as np
import struct

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

def read_fbin(filename, use_memmap=False):
    """读取fbin格式文件
    Args:
        filename: 文件路径
        use_memmap: 如果True，使用内存映射（适合大文件，不一次性加载到内存）
    """
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
    
    # 使用int64避免溢出：n * d * 4 可能非常大
    n = int(n)
    d = int(d)
    total_bytes = n * d * 4  # Python int 是任意精度的，不会溢出，但转换为int64确保
    
    if use_memmap:
        # 使用内存映射，从第8字节开始（跳过n和d）
        # 注意：np.memmap的shape参数需要是tuple，n和d需要是Python int
        vectors = np.memmap(filename, dtype=np.float32, mode='r', offset=8, shape=(n, d))
        return vectors
    else:
        # 一次性读取所有数据（仅用于小文件）
        # 注意：total_bytes可能很大，但f.read()可以处理任意大小的整数
        with open(filename, 'rb') as f:
            f.seek(8)  # 跳过n和d
            data_bytes = f.read(total_bytes)
            if len(data_bytes) != total_bytes:
                raise ValueError(f"数据不完整: 期望 {total_bytes} 字节，实际读取 {len(data_bytes)} 字节")
            vectors = np.frombuffer(data_bytes, dtype=np.float32).reshape(n, d)
        return vectors

def read_fbin_header(filename):
    """只读取fbin文件头（n和d），不读取数据"""
    with open(filename, 'rb') as f:
        n_bytes = f.read(4)
        if not n_bytes:
            raise ValueError("无法读取文件头: n")
        n = struct.unpack('i', n_bytes)[0]
        
        d_bytes = f.read(4)
        if not d_bytes:
            raise ValueError("无法读取文件头: d")
        d = struct.unpack('i', d_bytes)[0]
    
    # 转换为Python int（避免后续计算溢出）
    return int(n), int(d)

def read_bvecs_header(filename):
    """只读取bvecs文件的第一行来获取维度，不读取所有数据"""
    with open(filename, 'rb') as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            raise ValueError("无法读取文件头")
        dim = struct.unpack('i', dim_bytes)[0]
    return dim

def read_fvecs_header(filename):
    """只读取fvecs文件的第一行来获取维度，不读取所有数据"""
    with open(filename, 'rb') as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            raise ValueError("无法读取文件头")
        dim = struct.unpack('i', dim_bytes)[0]
    return dim

# 全局变量：用于缓存bvecs/fvecs文件的偏移量（避免重复计算）
_offset_cache = {}

def build_offset_cache(filename, n_vectors):
    """构建bvecs/fvecs文件的偏移量索引（只构建一次）"""
    if filename in _offset_cache:
        return _offset_cache[filename]
    
    print(f"  正在构建{filename}的偏移量索引（只需一次）...")
    offsets = []
    current_offset = 0
    
    with open(filename, 'rb') as f:
        for i in range(n_vectors):
            offsets.append(current_offset)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            if filename.endswith('.bvecs'):
                vector_size = 4 + dim  # 4字节dim + dim字节数据
            elif filename.endswith('.fvecs'):
                vector_size = 4 + dim * 4  # 4字节dim + dim*4字节float
            else:
                raise ValueError(f"不支持的文件格式: {filename}")
            current_offset += vector_size
            f.seek(dim if filename.endswith('.bvecs') else dim * 4, 1)
    
    _offset_cache[filename] = offsets
    print(f"  偏移量索引构建完成，共{n_vectors}个向量")
    return offsets

def read_base_vector_at_index(filename, index, d, dtype='float32'):
    """从base文件中读取指定索引的单个向量（按需读取）"""
    if filename.endswith('.fbin'):
        # fbin格式：前8字节是n和d，然后每个向量是d*4字节（固定大小，可以直接seek）
        # 使用int()确保不会溢出，Python int是任意精度的
        index = int(index)
        d = int(d)
        offset = 8 + index * d * 4
        vector_size = d * 4
        with open(filename, 'rb') as f:
            f.seek(offset)
            vector_bytes = f.read(vector_size)
            if len(vector_bytes) != vector_size:
                raise ValueError(f"读取向量失败，索引={index}")
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
    elif filename.endswith('.bvecs'):
        # bvecs格式：使用偏移量索引来快速定位
        if filename not in _offset_cache:
            # 需要先构建索引（需要知道向量总数，这里假设已经知道）
            # 如果不知道总数，可以从文件中统计（较慢但只需一次）
            pass  # 索引会在主程序中构建
        
        offsets = _offset_cache.get(filename)
        if offsets and index < len(offsets):
            with open(filename, 'rb') as f:
                f.seek(offsets[index])
                dim_bytes = f.read(4)
                dim = struct.unpack('i', dim_bytes)[0]
                vector_bytes = f.read(dim)
                vector = np.frombuffer(vector_bytes, dtype=np.uint8).astype(np.float32)
        else:
            # 如果没有索引，回退到遍历方式（很慢）
            with open(filename, 'rb') as f:
                for i in range(index + 1):
                    dim_bytes = f.read(4)
                    if not dim_bytes:
                        raise ValueError(f"索引{index}超出范围")
                    dim = struct.unpack('i', dim_bytes)[0]
                    if i == index:
                        vector_bytes = f.read(dim)
                        vector = np.frombuffer(vector_bytes, dtype=np.uint8).astype(np.float32)
                        break
                    else:
                        f.seek(dim, 1)
    elif filename.endswith('.fvecs'):
        # fvecs格式：使用偏移量索引来快速定位
        offsets = _offset_cache.get(filename)
        if offsets and index < len(offsets):
            with open(filename, 'rb') as f:
                f.seek(offsets[index])
                dim_bytes = f.read(4)
                dim = struct.unpack('i', dim_bytes)[0]
                vector_bytes = f.read(dim * 4)
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
        else:
            # 如果没有索引，回退到遍历方式
            with open(filename, 'rb') as f:
                for i in range(index + 1):
                    dim_bytes = f.read(4)
                    if not dim_bytes:
                        raise ValueError(f"索引{index}超出范围")
                    dim = struct.unpack('i', dim_bytes)[0]
                    if i == index:
                        vector_bytes = f.read(dim * 4)
                        vector = np.frombuffer(vector_bytes, dtype=np.float32)
                        break
                    else:
                        f.seek(dim * 4, 1)
    else:
        raise ValueError(f"不支持的文件格式: {filename}")
    
    return vector.astype(np.float32)

def read_ivecs(filename):
    """读取ivecs格式文件
    格式：每组先是一个k（4字节int），然后是k个邻居id（每个4字节int）
    """
    neighbors = []
    with open(filename, 'rb') as f:
        while True:
            k_bytes = f.read(4)
            if not k_bytes:
                break
            k = struct.unpack('i', k_bytes)[0]
            neighbor_bytes = f.read(k * 4)
            if not neighbor_bytes:
                break
            neighbor_ids = struct.unpack('i' * k, neighbor_bytes)
            neighbors.append(np.array(neighbor_ids, dtype=np.int32))
    return np.array(neighbors)

def read_vectors(filename, use_memmap=False):
    """读取向量文件，统一转换为float32格式
    Args:
        filename: 文件路径
        use_memmap: 对于fbin格式，是否使用内存映射（适合大文件）
    """
    if filename.endswith('.fbin'):
        vectors = read_fbin(filename, use_memmap=use_memmap)
    elif filename.endswith('.fvecs'):
        vectors = read_fvecs(filename)
    elif filename.endswith('.bvecs'):
        vectors = read_bvecs(filename)  # 读取为uint8
        # 转换为float32
        vectors = vectors.astype(np.float32)
    else:
        raise ValueError(f"未知文件格式: {filename}")
    
    # 确保返回float32格式（memmap不需要转换）
    if isinstance(vectors, np.memmap):
        return vectors
    return vectors.astype(np.float32)

def log(msg):
    print(msg)
    # with open(LOG_FILE, 'a') as lf:
    #     lf.write(str(msg) + "\n")
# ==========================
# 读取数据
# ==========================
# QUERY_PATH = "/data/myl/sift1M/sift1M_query.fvecs"
# BASE_PATH = "/data/myl/sift1M/sift1M_base.fvecs"
# GT_PATH = "/data/myl/sift1M/sift1M_groundtruth.ivecs"

QUERY_PATH = "/data/myl/sift1B/bigann_query.bvecs"
BASE_PATH = "/data/myl/sift1B/bigann_base.bvecs"
GT_PATH = "/data/myl/sift1B/gnd/idx_1000M.ivecs"
EPSILON = 1e-6  # 浮点数比较的容差

print("正在读取query向量...")
query_vectors = read_vectors(QUERY_PATH)
nq, d = query_vectors.shape
print(f"Query vectors shape: {query_vectors.shape}")

print("正在读取base向量元信息...")
# 只读取base文件的头信息，不加载所有数据
if BASE_PATH.endswith('.fbin'):
    nb, db = read_fbin_header(BASE_PATH)
    use_memmap = True  # 对于大文件使用memmap
elif BASE_PATH.endswith('.fvecs'):
    db = read_fvecs_header(BASE_PATH)
    # 需要统计数量并构建偏移量索引
    print("  正在统计base向量数量并构建偏移量索引...")
    nb = 0
    offsets = []
    current_offset = 0
    with open(BASE_PATH, 'rb') as f:
        while True:
            offsets.append(current_offset)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vector_size = 4 + dim * 4  # 4字节dim + dim*4字节float
            current_offset += vector_size
            f.seek(dim * 4, 1)  # 跳过向量数据
            nb += 1
    # 保存偏移量索引到缓存
    _offset_cache[BASE_PATH] = offsets
    print(f"  偏移量索引已构建，共{nb}个向量")
    use_memmap = False
elif BASE_PATH.endswith('.bvecs'):
    db = read_bvecs_header(BASE_PATH)
    # 需要统计数量并构建偏移量索引
    print("  正在统计base向量数量并构建偏移量索引...")
    nb = 0
    offsets = []
    current_offset = 0
    with open(BASE_PATH, 'rb') as f:
        while True:
            offsets.append(current_offset)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vector_size = 4 + dim  # 4字节dim + dim字节数据
            current_offset += vector_size
            f.seek(dim, 1)  # 跳过向量数据
            nb += 1
    # 保存偏移量索引到缓存
    _offset_cache[BASE_PATH] = offsets
    print(f"  偏移量索引已构建，共{nb}个向量")
    use_memmap = False
else:
    raise ValueError(f"不支持的文件格式: {BASE_PATH}")

print(f"Base向量数量: {nb}, 维度: {db}")

# 对于fbin格式的大文件，使用memmap；其他格式按需读取
if use_memmap:
    print("  使用内存映射方式访问base向量（不一次性加载到内存）...")
    base_vectors = read_fbin(BASE_PATH, use_memmap=True)
else:
    base_vectors = None  # 不加载所有数据，后续按需读取

print("正在读取groundtruth...")
gt_neighbors = read_ivecs(GT_PATH)
print(f"Groundtruth shape: {gt_neighbors.shape}")

# 检查维度是否匹配
if d != db:
    print(f"错误：维度不匹配！Query维度={d}, Base维度={db}")
    exit(1)

# 检查query数量和groundtruth数量是否匹配
if nq != len(gt_neighbors):
    print(f"错误：Query数量({nq})与Groundtruth数量({len(gt_neighbors)})不匹配！")
    exit(1)

# 确保query向量是float32格式
query_vectors = query_vectors.astype(np.float32)

# ==========================
# 比较每个query与其top1最近邻对应的base向量
# ==========================
print(f"\n正在比较每个query向量与其groundtruth中top1最近邻对应的base向量...")

# 统计结果
matched_count = 0
not_matched_count = 0
not_matched_indices = []
diffs = []

for i in range(nq):
    query_vec = query_vectors[i]
    
    # 获取groundtruth中的top1最近邻索引
    top1_idx = gt_neighbors[i][0]
    
    # 检查索引是否有效
    if top1_idx < 0 or top1_idx >= nb:
        print(f"警告：Query[{i}]的groundtruth索引{top1_idx}超出base范围[0, {nb-1}]")
        not_matched_count += 1
        not_matched_indices.append(i)
        continue
    
    # 获取对应的base向量（使用memmap或按需读取）
    if base_vectors is not None:
        # 使用memmap方式
        base_vec = base_vectors[top1_idx].astype(np.float32)
    else:
        # 按需读取
        base_vec = read_base_vector_at_index(BASE_PATH, top1_idx, db)
    
    # 比较query向量和base向量是否相同
    diff = np.abs(query_vec - base_vec)
    max_diff = np.max(diff)
    sum_diff = np.sum(diff)
    
    if max_diff < EPSILON:
        matched_count += 1
        if matched_count <= 10:
            print(f"  Query[{i}] 与base[{top1_idx}]完全匹配")
    else:
        not_matched_count += 1
        not_matched_indices.append(i)
        diffs.append((max_diff, sum_diff))
        if not_matched_count <= 10:
            print(f"  Query[{i}] 与base[{top1_idx}]不匹配（最大差异={max_diff:.8f}, 总差异={sum_diff:.8f}）")
    
    # 每处理1000个向量打印一次进度
    if (i + 1) % 1000 == 0:
        print(f"  已处理 {i + 1}/{nq} 个query向量...")

# ==========================
# 输出结果
# ==========================
print("\n" + "="*60)
print("验证结果：")
print("="*60)
print(f"Query向量总数: {nq}")
print(f"与groundtruth top1对应的base向量完全匹配: {matched_count} ({matched_count/nq*100:.2f}%)")
print(f"与groundtruth top1对应的base向量不匹配: {not_matched_count} ({not_matched_count/nq*100:.2f}%)")

if matched_count == nq:
    print("\n✓ 所有query向量都与其groundtruth top1最近邻对应的base向量相同！")
else:
    print(f"\n✗ 有 {not_matched_count} 个query向量与其groundtruth top1最近邻对应的base向量不同")
    if len(not_matched_indices) > 10:
        print(f"  （仅显示前10个，共{len(not_matched_indices)}个）")

# 额外检查：查看不匹配向量的差异分布
if not_matched_count > 0 and len(diffs) > 0:
    print(f"\n未匹配向量的差异统计：")
    max_diffs = [d[0] for d in diffs]
    sum_diffs = [d[1] for d in diffs]
    max_diffs = np.array(max_diffs)
    sum_diffs = np.array(sum_diffs)
    
    print(f"最大差异统计（每个维度）：")
    print(f"  最小值: {max_diffs.min():.8f}")
    print(f"  最大值: {max_diffs.max():.8f}")
    print(f"  平均值: {max_diffs.mean():.8f}")
    print(f"  中位数: {np.median(max_diffs):.8f}")
    
    print(f"\n总差异统计（所有维度之和）：")
    print(f"  最小值: {sum_diffs.min():.8f}")
    print(f"  最大值: {sum_diffs.max():.8f}")
    print(f"  平均值: {sum_diffs.mean():.8f}")
    print(f"  中位数: {np.median(sum_diffs):.8f}")

