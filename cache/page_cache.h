#pragma once

#include "cache/lru.h"
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../head.h"

/**
 * 点的信息结构
 */
struct PointInfo {
    // int belong;          // 所属的cluster_id
    // int local_page_id;   // 在cluster内的page_id
    int offset;          // 在page内的偏移
    int global_page_id;  // 在全局page列表中的ID

    __host__ __device__ PointInfo() : offset(-1), global_page_id(-1) {}
    __host__ __device__ PointInfo(int off, int gpid)
        : offset(off), global_page_id(gpid) {}
};

// struct PointInfo {
//     int belong;          // 所属的cluster_id
//     int local_page_id;   // 在cluster内的page_id
//     int offset;          // 在page内的偏移
//     int global_page_id;  // 在全局page列表中的ID

//     __host__ __device__ PointInfo() : belong(-1), local_page_id(-1), offset(-1), global_page_id(-1) {}
//     __host__ __device__ PointInfo(int b, int lpid, int off, int gpid)
//         : belong(b), local_page_id(lpid), offset(off), global_page_id(gpid) {}
// };

struct PointInfo_local {
    int belong;          // 所属的cluster_id
    int local_page_id;   // 在cluster内的page_id

    __host__ __device__ PointInfo_local() : belong(-1), local_page_id(-1) {}
    __host__ __device__ PointInfo_local(int b, int lpid)
        : belong(b), local_page_id(lpid) {}
};

/**
 * Page Cache 管理器
 */
class PageCache {
private:
    // Cache基本属性
    int page_size;           // 每个page包含的点数
    int num_pages;           // cache中的page总数
    int dim_partial;         // 后(D-d)维的维度
    int num_points_total;    // 总点数
    
    // 数据存储
    float* cache_data;           // cache数据（GPU内存），大小为 num_pages * page_size * dim_partial
    float* full_data;            // 完整数据（CPU固定页内存），按cluster排序
    
    // 映射关系
    int num_clusters;        // 聚类总数
    // std::vector<int> cluster_page_count;    // 每个cluster真实的page数量
    std::vector<int> cluster_page_offset;   // cluster在全局page数组中的前缀和
    // int total_cluster_pages;                // 全部cluster的page总数
    // std::vector<int> cluster_to_page;       // cluster page到cache page的映射，大小为 total_cluster_pages
    std::vector<PointInfo> point_info; // 点id到其信息的映射
    std::vector<PointInfo_local> point_info_local;

    // 设备端辅助结构
    int* device_cluster_to_page;        // 设备端映射
    bool constant_cluster_map_enabled;
    PointInfo* device_point_info;       // 设备端点信息
    
    // LRU管理
    LRUList* lru;
    cudaStream_t default_stream;
    bool owns_stream;
    
    // 统计信息
    long long cache_hits;
    long long cache_misses;

public:
    // std::vector<int> cluster_to_page;       // cluster page到cache page的映射，大小为 total_cluster_pages
    int* cluster_to_page;
    int total_cluster_pages;                // 全部cluster的page总数
    std::vector<int> cluster_page_count;    // 每个cluster真实的page数量
    // 拷贝页数
    long long copied_pages;
    
public:
    /**
     * 构造函数
     * @param page_size 每个page包含的点数
     * @param num_pages cache中的page总数
     * @param dim_partial 后(D-d)维的维度
     * @param num_clusters 聚类总数
     * @param num_points 总点数
     */
    PageCache(int page_size, int num_pages, int dim_partial, int num_clusters, int num_points);
    
    /**
     * 析构函数
     */
    ~PageCache();
    
    /**
     * 初始化完整数据（按cluster排序的点的后(D-d)维坐标）
     * @param data 数据指针，大小为 num_points * dim_partial
     * @param row_stride 数据行步长（例如，如果数据是连续的，则为dim_partial）
     * @param value_offset 数据值的偏移量（例如，如果数据包含其他信息，则为0）
     * @param cluster_labels 点到cluster的映射，大小为 num_points
     * @param cluster_points 每个cluster包含的点ID列表
     */
    void init_data(const float* data,
                   int row_stride,
                   int value_offset,
                   const std::vector<int>& cluster_labels,
                   const std::vector<std::vector<int>>& cluster_points);
    
    /**
     * 获取点的缓存数据指针
     * @param point_id 点的ID
     * @return 如果cache hit，返回指向cache中该点数据的指针；否则返回nullptr
     */
    float* get_point(int point_id, cudaStream_t stream = nullptr);
    
    /**
     * 预加载指定cluster的所有page到cache
     * @param cluster_id 要预加载的cluster_id
     */
    void prefetch_cluster(int cluster_id, int& copy_count, cudaStream_t stream = nullptr);

    /**
     * 按批次预加载多个cluster的所有page（使用同一个stream异步拷贝）
     */
    void prefetch_clusters_async(const std::vector<int>& cluster_ids, int& copy_count, cudaStream_t stream = nullptr);

    /**
     * 将指定点的数据拷贝回主机内存（便于调试）
     */
    void copy_point_to_host(int point_id, float* host_buffer, cudaStream_t stream = nullptr);

    /**
     * 获取设备端指针，便于GPU核函数直接访问
     */
    float* device_cache_ptr() const { return cache_data; }
    const int* device_cluster_map() const { return device_cluster_to_page; }
    const PointInfo* device_point_info_ptr() const { return device_point_info; }
    bool using_constant_cluster_map() const { return constant_cluster_map_enabled; }

    /**
     * 获取内部默认stream
     */
    cudaStream_t get_default_stream() const { return default_stream; }
    
    /**
     * 获取统计信息
     */
    void print_stats() const;
    long long get_cache_hits() const { return cache_hits; }
    long long get_cache_misses() const { return cache_misses; }
    double get_hit_rate() const { 
        long long total = cache_hits + cache_misses;
        return total > 0 ? (double)cache_hits / total : 0.0;
    }
    
    /**
     * 重置统计信息
     */
    void reset_stats() { cache_hits = 0; cache_misses = 0; }
    
    /**
     * 获取cache基本信息
     */
    int get_page_size() const { return page_size; }
    int get_num_pages() const { return num_pages; }
    int get_dim_partial() const { return dim_partial; }
    
    /**
     * 从global_page_id=0开始顺序填充cache
     */
    void random_fill_cache();

    /**
     * 更新cluster_to_page映射
     */
    void update_map(cudaStream_t stream = nullptr);
private:
    /**
     * 加载指定page到cache
     * @param cluster_id cluster的ID
     * @param local_page_id 在cluster内的page_id
     * @return cache中该page的起始地址
     */
    float* load_page(int cluster_id, int local_page_id, cudaStream_t stream);
    
    /**
     * 计算global page_id
     */
    // int get_global_page_id(int cluster_id, int local_page_id) const;
    
    /**
     * 获取点在full_data中的起始索引
     */
    int get_point_index_in_full_data(int cluster_id, int local_page_id, int offset) const;

    cudaStream_t resolve_stream(cudaStream_t stream) const;

public:
    int get_global_page_id(int cluster_id, int local_page_id) const;
};

