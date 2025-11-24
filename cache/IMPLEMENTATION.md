# Page Cache System 实现总结

## 📁 文件结构

```
cache/
├── lru.h                   # LRU 双向链表头文件
├── lru.cpp                 # LRU 双向链表实现
├── page_cache.h            # Page Cache 管理器头文件
├── page_cache.cpp          # Page Cache 管理器实现
├── test_cache.cpp          # 测试程序
├── Makefile                # 编译脚本
├── README.md               # 使用说明
└── IMPLEMENTATION.md       # 本文件（实现总结）
```

## 🎯 核心数据结构

### 1. LRUNode (lru.h)
```cpp
struct LRUNode {
    int page_id;           // page在cache中的序号
    int cluster_id;        // page所属的cluster_id
    int local_page_id;     // 在cluster内部的page_id
    LRUNode* next;         // 指向下一节点
    LRUNode* prev;         // 指向上一节点
};
```

### 2. LRUList (lru.h/cpp)
**功能**: 管理 LRU 双向链表

**核心方法**:
- `void touch(int page_id)`: 将 page 移到链表头部
- `int insert(cluster_id, local_page_id, old_cluster_id, old_local_page_id)`: 
  - 从尾部淘汰 page
  - 插入新 page 到头部
  - 返回被淘汰的 page_id

**实现细节**:
- 使用哨兵节点 (head/tail) 简化边界处理
- 维护 `nodes[]` 数组实现 O(1) 的 page_id 到节点的查找

### 3. PointInfo (page_cache.h)
```cpp
struct PointInfo {
    int belong;          // 所属的cluster_id
    int local_page_id;   // 在cluster内的page_id
    int offset;          // 在page内的偏移
};
```

### 4. PageCache (page_cache.h/cpp)
**功能**: 管理整个 cache 系统

**核心属性**:
```cpp
int page_size;                        // 每个page包含的点数
int num_pages;                        // cache中的page总数
int dim_partial;                      // 后(D-d)维的维度
float* cache_data;                    // cache数据
float* full_data;                     // 完整数据（按cluster排序）
vector<int> cluster_to_page;          // cluster page → cache page映射
vector<PointInfo> point_info;         // 点id → 点信息映射
LRUList* lru;                         // LRU管理器
```

**核心方法**:
- `void init_data(data, row_stride, value_offset, ...)`: 初始化数据，按 cluster 重新排序
- `float* get_point(point_id)`: 获取点的数据指针
  - 如果 cache hit: 返回 cache 中的指针，更新 LRU
  - 如果 cache miss: 加载 page，返回指针
- `void prefetch_cluster(cluster_id)`: 预加载整个 cluster 的所有 page
- `void print_stats()`: 打印统计信息

## 🔄 工作流程

### 初始化流程
```
1. 创建 PageCache 对象
   ↓
2. 调用 init_data(data, dim_partial, 0, cluster_labels, cluster_points)
   ↓
3. 按 cluster 重新排序数据 → full_data
   ↓
4. 计算每个点的 PointInfo (belong, local_page_id, offset)
   ↓
5. 初始化 cluster_to_page 映射（全部为-1）
```

### 访问点的流程
```
get_point(point_id)
   ↓
获取 PointInfo: belong, local_page_id, offset
   ↓
计算 global_page_id = belong * pages_per_cluster + local_page_id
   ↓
查询 cluster_to_page[global_page_id]
   ↓
   ├─ == -1 (Cache Miss)
   │    ↓
   │  load_page(belong, local_page_id)
   │    ↓
   │  LRU.insert() → 获取 victim_page_id
   │    ↓
   │  更新 cluster_to_page 映射
   │    ↓
   │  从 full_data 复制数据到 cache
   │    ↓
   │  返回 cache 中的指针
   │
   └─ != -1 (Cache Hit)
        ↓
      LRU.touch(cache_page_id)
        ↓
      返回 cache 中的指针
```

## 📊 映射关系详解

### 1. cluster page → cache page 映射
```cpp
// 计算 global page id
global_page_id = cluster_id * pages_per_cluster + local_page_id

// 查询映射
cache_page_id = cluster_to_page[global_page_id]

// -1 表示不在 cache 中
if (cache_page_id == -1) {
    // cache miss
} else {
    // cache hit
    address = cache_data + cache_page_id * page_size * dim_partial
}
```

### 2. 点id → 点信息映射
```cpp
PointInfo info = point_info[point_id]
// info.belong: 所属cluster
// info.local_page_id: 在cluster内的page编号
// info.offset: 在page内的偏移
```

### 3. 点id → cache地址映射
```cpp
float* get_point(int point_id) {
    PointInfo info = point_info[point_id];
    int global_page_id = get_global_page_id(info.belong, info.local_page_id);
    int cache_page_id = cluster_to_page[global_page_id];
    
    if (cache_page_id == -1) {
        // cache miss - load page
        float* page_addr = load_page(info.belong, info.local_page_id);
        return page_addr + info.offset * dim_partial;
    } else {
        // cache hit
        lru->touch(cache_page_id);
        return cache_data + cache_page_id * page_size * dim_partial 
                          + info.offset * dim_partial;
    }
}
```

## 🧪 测试验证

### 测试1: 基本功能 ✅
- Cache miss/hit 正确
- 数据读取正确
- LRU 淘汰正确

### 测试2: Prefetch ✅
- 预加载功能正常
- 预加载后命中率 100%

### 测试3: LRU 行为 ✅
- 最久未使用的 page 被正确淘汰
- LRU 顺序维护正确

## 📈 性能分析

### 时间复杂度
- `get_point()`: O(1) - 直接通过映射表查找
- `touch()`: O(1) - 链表操作
- `insert()`: O(1) - 链表操作
- `prefetch_cluster()`: O(pages_per_cluster) - 线性加载

### 空间复杂度
- cache 数据: O(num_pages × page_size × dim_partial)
- full 数据: O(num_points × dim_partial)
- 映射表: O(num_clusters × pages_per_cluster)
- LRU 链表: O(num_pages)

### 优化点
1. **空间局部性**: 同 cluster 的点在内存中连续
2. **时间局部性**: LRU 保证热数据在 cache
3. **批量加载**: 以 page 为单位减少加载次数
4. **O(1) 访问**: 无需搜索，直接定位

## 🔧 编译和使用

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

### 集成到项目
```cpp
#include "cache/page_cache.h"

PageCache cache(page_size, num_pages, dim_partial, num_clusters, num_points);
cache.init_data(data, dim_partial, 0, cluster_labels, cluster_points);

// 访问点
float* point_data = cache.get_point(point_id);
```

## ✨ 特性

✅ 完整的 LRU 替换策略  
✅ O(1) 访问时间复杂度  
✅ 支持预加载整个 cluster  
✅ 详细的统计信息  
✅ 清晰的代码结构  
✅ 完整的测试覆盖  

## 🚀 未来扩展

1. **GPU 版本**: 实现 GPU cache，利用高带宽内存
2. **智能预取**: 基于访问模式的预取策略
3. **压缩存储**: 对 page 进行压缩节省空间
4. **多级 cache**: L1/L2 多级 cache 结构
5. **并行访问**: 多线程安全的 cache 实现
6. **自适应**: 根据命中率动态调整 cache 大小

## 📝 总结

本实现提供了一个**完整、高效、易用**的 Page Cache 系统：

- **完整性**: 包含 LRU、映射、统计等所有必要组件
- **高效性**: O(1) 访问，最优的时间复杂度
- **易用性**: 清晰的 API，详细的文档和测试
- **可扩展性**: 模块化设计，易于扩展和集成

所有代码已经过测试验证，可以直接使用！✅

