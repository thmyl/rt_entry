# Page Cache System

基于 Page 的 Cache 存储系统，用于管理点的后 (D-d) 维坐标，采用 LRU 替换策略。

## 设计概述

### 核心思想
- **Page-based Storage**: 以 page 为单位存储和管理数据
- **Cluster Organization**: 数据按 cluster 分组，同一 cluster 的数据在内存中连续存储
- **LRU Replacement**: 采用 LRU (Least Recently Used) 策略进行 page 替换

### 数据结构

#### 1. LRUList (lru.h/cpp)
双向链表实现的 LRU 管理器：
- **touch(page_id)**: 将 page 移动到链表头部（标记为最近使用）
- **insert(cluster_id, local_page_id)**: 淘汰尾部 page，插入新 page 到头部

#### 2. PageCache (page_cache.h/cpp)
Cache 管理器，核心组件：

**属性**:
- `page_size`: 每个 page 包含的点数
- `num_pages`: cache 可容纳的 page 总数
- `dim_partial`: 后 (D-d) 维的维度

**映射关系**:
1. **cluster_page_offset**: 前缀和数组，用于将 `(cluster_id, local_page_id)` 映射到全局 `global_page_id`
2. **cluster_to_page**: `global_page_id → cache_page_id` 的映射
   - 值为 -1 表示不在 GPU cache 中
3. **point_info**: 点 ID → 点信息
   - `belong`: 所属的 cluster_id
   - `local_page_id`: 在 cluster 内的 page_id
   - `offset`: 在 page 内的偏移
   - `global_page_id`: 预先计算好的全局 page ID，便于 GPU 端快速索引

## 文件说明

- `lru.h/cpp`: LRU 双向链表实现
- `page_cache.h/cpp`: Page Cache 管理器实现
- `test_cache.cpp`: 测试程序
- `Makefile`: 编译脚本

## 编译和测试

### 编译
```bash
cd cache_search/cache
make
```

### 运行测试
```bash
make test
```

### 清理
```bash
make clean
```

## 使用示例

```cpp
#include "page_cache.h"
// 假设已定义 CUDA_CHECK(...) 辅助宏

// 1. 创建 cache
int page_size = 100;        // 每个 page 100 个点
int num_pages = 10;         // cache 容纳 10 个 page
int dim_partial = 96;       // 后 96 维
int num_clusters = 100;     // 100 个 cluster
int num_points = 1000000;   // 100 万个点

PageCache cache(page_size, num_pages, dim_partial, num_clusters, num_points);

// 2. 初始化数据
float* data = ...;  // 点的后 (D-d) 维坐标
std::vector<int> cluster_labels = ...;  // 点到 cluster 的映射
std::vector<std::vector<int>> cluster_points = ...;  // 每个 cluster 包含的点

cache.init_data(data, dim_partial, 0, cluster_labels, cluster_points);

cudaStream_t stream;
CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

// 3. 预加载一批 cluster 到 GPU cache（异步）
std::vector<int> clusters = {0, 5, 7};
cache.prefetch_clusters_async(clusters, stream);

// 4. 将某个点的数据拷回主机做验证（真实使用时直接在 GPU 上访问）
std::vector<float> host_buffer(dim_partial);
int point_id = cluster_points[clusters.front()][0]; // 取 batch 中的一个点
cache.copy_point_to_host(point_id, host_buffer.data(), stream);
CUDA_CHECK(cudaStreamSynchronize(stream));

// 5. 查看统计
cache.print_stats();

CUDA_CHECK(cudaStreamDestroy(stream));
```

## 测试说明

测试程序通过模拟若干 query batch，展示：
- 使用 `prefetch_clusters_async` 在 GPU stream 上批量加载 cluster
- GPU Cache 命中率统计
- 未预取情况下的按需加载与 LRU 淘汰

## 性能特点

1. **空间局部性**: 同一 cluster 的点在内存中连续，利于缓存
2. **时间局部性**: LRU 策略保证最近访问的数据留在 cache
3. **批量加载**: 支持在任意 CUDA stream 上异步预加载 cluster，减少 miss
4. **O(1) 访问**: 通过映射表直接定位，GPU kernel 可直接使用 `global_page_id`

## 统计信息

Cache 提供以下统计：
- `cache_hits`: 命中次数
- `cache_misses`: 未命中次数
- `hit_rate`: 命中率 = hits / (hits + misses)

## 内存布局

```
Full Data (按 cluster 排序):
[Cluster 0: Page 0 | Page 1 | ... | Page N]
[Cluster 1: Page 0 | Page 1 | ... | Page N]
...

Cache (只保存部分 page):
[Page 0] [Page 1] ... [Page num_pages-1]

LRU 链表 (从头到尾表示从最近到最久):
Head -> [Page 3] -> [Page 7] -> [Page 1] -> ... -> Tail
```

## 注意事项

1. **Page 对齐**: 每个 cluster 的最后一个 page 如果不满，会用 0 填充
2. **异步拷贝**: 使用 `cudaMemcpyAsync`，若需与 CPU 同步，需手动 `cudaStreamSynchronize`
3. **内存占用**: full_data 使用固定页内存（pinned memory）以支持异步拷贝
4. **线程安全**: 当前实现不是线程安全的，多线程需要加锁

## 扩展建议

1. **预取策略**: 可以根据访问模式智能预取
2. **压缩存储**: 可以对 page 进行压缩以节省空间
3. **多级 cache**: 可以实现 L1/L2 多级 cache 结构
4. **统计工具**: 增加更详细的访问与带宽统计

