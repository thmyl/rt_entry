# Page Cache 快速开始指南

## 🚀 5分钟快速上手

> ⚠️ 说明：当前实现将 cache 数据存放在 GPU 上，`get_point()` 返回设备指针，只能在 GPU kernel 中直接使用。若需在 CPU 端调试/打印，请改用 `copy_point_to_host()`。

### 1. 编译测试
```bash
cd /home/myl/cache_search/cache
make test
```

### 2. 基本使用示例

```cpp
#include "page_cache.h"
#include <cuda_runtime.h>
#include <vector>

int main() {
    // === 第1步：准备数据 ===
    int num_points = 1000000;        // 100万个点
    int num_clusters = 100;          // 100个cluster
    int dim_partial = 96;            // 后96维
    
    // 点的后(D-d)维坐标数据
    float* data = new float[num_points * dim_partial];
    // ... 填充数据 ...
    
    // 点到cluster的映射
    std::vector<int> cluster_labels(num_points);
    // ... 从聚类结果读取 ...
    
    // 每个cluster包含的点列表
    std::vector<std::vector<int>> cluster_points(num_clusters);
    for (int i = 0; i < num_points; i++) {
        cluster_points[cluster_labels[i]].push_back(i);
    }
    
    // === 第2步：创建Cache ===
    int page_size = 100;             // 每个page 100个点
    int num_pages = 50;              // cache容纳50个page
    
    PageCache cache(page_size, num_pages, dim_partial, 
                   num_clusters, num_points);
    
    // === 第3步：初始化数据 ===
    cache.init_data(data, dim_partial, 0, cluster_labels, cluster_points);
    
    // === 第4步：使用Cache ===
    
    // 4.1 预加载cluster
    int cluster_id = 5;
    cache.prefetch_cluster(cluster_id);
    
    // 4.2 从GPU cache中取出一个点到CPU做验证
    int point_id = cluster_points[cluster_id].front();
    std::vector<float> host_buffer(dim_partial);
    cache.copy_point_to_host(point_id, host_buffer.data());
    cudaStreamSynchronize(cache.get_default_stream());
    // host_buffer 现在包含点的 dim_partial 维数据
    
    // 4.3 实际使用时，在GPU kernel中通过 point_info/global_page_id 定位
    auto device_cache = cache.device_cache_ptr();
    auto device_point_meta = cache.device_point_info_ptr();
    auto device_page_map = cache.device_cluster_map();
    
    // === 第5步：查看统计 ===
    cache.print_stats();
    
    delete[] data;
    return 0;
}
```

## 📊 关键参数选择

### page_size（每个page包含的点数）
- **太小**: 频繁加载，overhead大
- **太大**: 浪费cache空间
- **推荐**: 
  - 小数据集: 50-100
  - 大数据集: 100-500
  - 考虑 cache line 大小（通常64字节）

### num_pages（cache中的page总数）
- **计算**: `cache_size_MB = num_pages × page_size × dim_partial × 4 / 1024 / 1024`
- **推荐**: 
  - 根据可用内存: 1GB ~ 10GB
  - 根据访问模式: 如果访问集中在少数cluster，可以更小
  - 示例: `num_pages = 1000, page_size = 100, dim = 96 → 约 36MB`

### 命中率预期
- **随机访问**: 低命中率 (< 20%)
- **cluster内访问**: 高命中率 (> 80%)
- **预加载后访问**: 接近100%命中率

## 🎯 常见使用场景

### 场景1: 搜索最近邻（按cluster访问）
```cpp
// 预加载最近的t个cluster
for (int i = 0; i < t; i++) {
    int nearest_cluster = nearest_clusters[i];
    cache.prefetch_cluster(nearest_cluster);
}

// 访问这些cluster中的点
for (int i = 0; i < t; i++) {
    for (int pid : cluster_points[nearest_clusters[i]]) {
        float* data = cache.get_point(pid);
        // 计算距离...
    }
}

cache.print_stats();  // 应该有很高的命中率
```

### 场景2: 随机访问（测试worst case）
```cpp
cache.reset_stats();

// 随机访问点
for (int i = 0; i < 1000; i++) {
    int pid = rand() % num_points;
    float* data = cache.get_point(pid);
    // 处理...
}

cache.print_stats();  // 命中率会较低
```

### 场景3: 批量计算距离
```cpp
void compute_distances(PageCache& cache, 
                      const std::vector<int>& candidate_points,
                      const float* query,
                      int dim) {
    for (int pid : candidate_points) {
        float* point_data = cache.get_point(pid);
        
        // 计算距离
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = query[d] - point_data[d];
            dist += diff * diff;
        }
        
        // 处理距离...
    }
}
```

## 🔍 调试技巧

### 1. 检查命中率
```cpp
cache.print_stats();

// 期望:
// - 随机访问: 10-20%
// - cluster内访问: 80-95%
// - 预加载后: 95-100%
```

### 2. 验证数据正确性
```cpp
// 从cache读取
float* cached_data = cache.get_point(point_id);

// 从原始数据读取
float* original_data = data + point_id * dim_partial;

// 比较
bool correct = true;
for (int d = 0; d < dim_partial; d++) {
    if (fabs(cached_data[d] - original_data[d]) > 1e-6) {
        correct = false;
        break;
    }
}
```

### 3. 性能测试
```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

// 执行访问...
for (int i = 0; i < N; i++) {
    cache.get_point(i);
}

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

std::cout << "访问 " << N << " 个点耗时: " << duration.count() << " ms" << std::endl;
std::cout << "平均每个点: " << (double)duration.count() / N << " ms" << std::endl;
```

## ⚠️ 注意事项

### 1. 内存管理
- PageCache 会分配 `full_data` 和 `cache_data`
- 析构函数会自动释放内存
- 不要手动 delete cache 返回的指针

### 2. 指针有效性
```cpp
float* p1 = cache.get_point(100);
// ... 使用 p1 ...

float* p2 = cache.get_point(200);  // 可能导致page被淘汰
// ⚠️ 此时 p1 可能失效！
```

**安全做法**:
```cpp
float* p1 = cache.get_point(100);
// 立即使用或复制数据
float value = p1[0];
```

### 3. 线程安全
- 当前实现**不是**线程安全的
- 多线程访问需要加锁

### 4. cluster大小不均
- 如果 cluster 大小差异很大，某些 cluster 可能占用很多 page
- 考虑限制每个 cluster 的最大 page 数

## 📈 性能优化建议

### 1. 访问模式优化
```cpp
// ❌ 差的访问模式（跨cluster随机访问）
for (int i = 0; i < N; i++) {
    int pid = random_points[i];
    process(cache.get_point(pid));
}

// ✅ 好的访问模式（按cluster分组）
for (int cluster_id : relevant_clusters) {
    cache.prefetch_cluster(cluster_id);
    for (int pid : cluster_points[cluster_id]) {
        process(cache.get_point(pid));
    }
}
```

### 2. 预加载策略
```cpp
// 如果知道将要访问的cluster列表
std::vector<int> to_visit = get_relevant_clusters();

// 预加载所有
for (int cid : to_visit) {
    cache.prefetch_cluster(cid);
}

// 然后访问（高命中率）
for (int cid : to_visit) {
    for (int pid : cluster_points[cid]) {
        process(cache.get_point(pid));
    }
}
```

### 3. Cache大小调优
```cpp
// 测试不同的cache大小
for (int num_pages : {10, 50, 100, 200, 500}) {
    PageCache cache(page_size, num_pages, dim, num_clusters, num_points);
    cache.init_data(...);
    
    // 运行测试
    run_test(cache);
    
    // 比较命中率和性能
    std::cout << "num_pages=" << num_pages 
              << " hit_rate=" << cache.get_hit_rate() << std::endl;
}
```

## 📞 获取帮助

- 查看 `README.md` 了解详细说明
- 查看 `IMPLEMENTATION.md` 了解实现细节
- 运行 `make test` 查看测试示例
- 查看 `test_cache.cpp` 了解更多使用示例

## ✅ 快速检查清单

- [ ] 编译通过 (`make`)
- [ ] 测试通过 (`make test`)
- [ ] 了解基本 API (get_point, prefetch_cluster)
- [ ] 选择合适的 page_size 和 num_pages
- [ ] 按 cluster 组织访问模式
- [ ] 监控命中率（prefetch后应 >80%）

现在你已经准备好使用 Page Cache 了！🎉

