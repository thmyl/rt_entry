#include "page_cache.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA 调用失败: " << expr << " (" << cudaGetErrorString(err)
            << ") at " << file << ":" << line;
        throw std::runtime_error(oss.str());
    }
}

} // anonymous namespace

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

PageCache::PageCache(int page_size, int num_pages, int dim_partial, int num_clusters, int num_points)
    : page_size(page_size),
      num_pages(num_pages),
      dim_partial(dim_partial),
      num_points_total(num_points),
      cache_data(nullptr),
      full_data(nullptr),
      num_clusters(num_clusters),
      total_cluster_pages(0),
      device_cluster_to_page(nullptr),
      device_point_info(nullptr),
      lru(nullptr),
      default_stream(nullptr),
      owns_stream(true),
      cache_hits(0),
      cache_misses(0) {

    if (page_size <= 0 || num_pages <= 0 || dim_partial <= 0) {
        throw std::invalid_argument("PageCache 参数必须为正数");
    }

    // 分配GPU cache空间
    size_t cache_elems = static_cast<size_t>(num_pages) * page_size * dim_partial;
    CUDA_CHECK(cudaMalloc(&cache_data, cache_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(cache_data, 0, cache_elems * sizeof(float)));

    // 分配设备端PointInfo
    if (num_points_total > 0) {
        CUDA_CHECK(cudaMalloc(&device_point_info, num_points_total * sizeof(PointInfo)));
    }

    // 初始化point_info
    point_info.resize(num_points_total);

    // 创建LRU管理器
    lru = new LRUList(num_pages);

    // 创建内部stream
    CUDA_CHECK(cudaStreamCreateWithFlags(&default_stream, cudaStreamNonBlocking));
}

PageCache::~PageCache() {
    if (owns_stream && default_stream != nullptr) {
        cudaStreamDestroy(default_stream);
    }

    if (cache_data) {
        cudaFree(cache_data);
    }
    if (full_data) {
        cudaFreeHost(full_data);
    }
    if (device_cluster_to_page) {
        cudaFree(device_cluster_to_page);
    }
    if (device_point_info) {
        cudaFree(device_point_info);
    }

    delete lru;
}

void PageCache::init_data(const float* data,
                         int row_stride,
                         int value_offset,
                         const std::vector<int>& cluster_labels,
                         const std::vector<std::vector<int>>& cluster_points) {
    int num_points = cluster_labels.size();

    if (num_points != num_points_total) {
        throw std::runtime_error("init_data: 输入点数与构造时不一致");
    }
    if (row_stride <= 0) {
        throw std::invalid_argument("init_data: row_stride must be positive");
    }
    if (value_offset < 0 || value_offset + dim_partial > row_stride) {
        throw std::out_of_range("init_data: value_offset 超出范围");
    }
    
    // 计算每个cluster需要的page数量
    int max_cluster_size = 0;
    cluster_page_count.assign(num_clusters, 0);
    cluster_page_offset.assign(num_clusters + 1, 0);
    for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
        const auto& pts = cluster_points[cluster_id];
        max_cluster_size = std::max(max_cluster_size, static_cast<int>(pts.size()));
        int page_cnt = pts.empty() ? 0 : (static_cast<int>(pts.size()) + page_size - 1) / page_size;
        cluster_page_count[cluster_id] = page_cnt;
        cluster_page_offset[cluster_id + 1] = cluster_page_offset[cluster_id] + page_cnt;
    }
    total_cluster_pages = cluster_page_offset[num_clusters];
    int max_pages_per_cluster = max_cluster_size == 0 ? 0 : (max_cluster_size + page_size - 1) / page_size;
    
    std::cout << "初始化PageCache:" << std::endl;
    std::cout << "  点数: " << num_points << std::endl;
    std::cout << "  聚类数: " << num_clusters << std::endl;
    std::cout << "  每个cluster最大page数: " << max_pages_per_cluster << std::endl;
    std::cout << "  总page数(实际): " << total_cluster_pages << std::endl;
    std::cout << "  cache容量: " << num_pages << " pages" << std::endl;
    
    // 初始化cluster_to_page映射（初始全部为-1，表示不在cache中）
    cluster_to_page.assign(total_cluster_pages, -1);
    printf("total_cluster_pages = %d\n", total_cluster_pages);
    
    if (device_cluster_to_page) {
        CUDA_CHECK(cudaFree(device_cluster_to_page));
    }
    if (total_cluster_pages > 0) {
        CUDA_CHECK(cudaMalloc(&device_cluster_to_page, total_cluster_pages * sizeof(int)));
        CUDA_CHECK(cudaMemset(device_cluster_to_page, 0xFF, total_cluster_pages * sizeof(int)));
    } else {
        device_cluster_to_page = nullptr;
    }

    // 分配full_data并按cluster重新排序
    size_t total_slots = static_cast<size_t>(total_cluster_pages) * page_size;
    size_t full_data_elems = total_slots * dim_partial;
    if (full_data) {
        CUDA_CHECK(cudaFreeHost(full_data));
        full_data = nullptr;
    }

    if (full_data_elems > 0) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&full_data), full_data_elems * sizeof(float)));
        memset(full_data, 0, full_data_elems * sizeof(float));
    }
    
    for (int cluster_id = 0; cluster_id < num_clusters; cluster_id++) {
        const auto& pts = cluster_points[cluster_id];

        int base_index = cluster_page_offset[cluster_id] * page_size;
        for (size_t i = 0; i < pts.size(); i++) {
            int point_id = pts[i];
            int local_page_id = i / page_size;
            int offset = i % page_size;
            int global_page_id = get_global_page_id(cluster_id, local_page_id);

            // 设置point_info
            point_info[point_id] = PointInfo(cluster_id, local_page_id, offset, global_page_id);

            // 复制数据到full_data
            if (full_data) {
                const float* src = data + static_cast<size_t>(point_id) * row_stride + value_offset;
                memcpy(full_data + static_cast<size_t>(base_index + i) * dim_partial,
                       src,
                       dim_partial * sizeof(float));
            }
        }

        // 其余补零的区域在上面的memset中已覆盖
    }
    
    if (num_points_total > 0 && device_point_info) {
        CUDA_CHECK(cudaMemcpy(device_point_info, point_info.data(),
                              static_cast<size_t>(num_points_total) * sizeof(PointInfo),
                              cudaMemcpyHostToDevice));
    }

    std::cout << "  数据初始化完成" << std::endl;
}

float* PageCache::get_point(int point_id, cudaStream_t stream) {
    cudaStream_t use_stream = resolve_stream(stream);

    const PointInfo& info = point_info[point_id];
    int global_page_id = info.global_page_id;
    
    // 检查是否在cache中
    int cache_page_id = cluster_to_page[global_page_id];
    
    if (cache_page_id == -1) {
        // Cache miss - 加载page
        cache_misses++;
        float* page_addr = load_page(info.belong, info.local_page_id, use_stream);
        return page_addr + info.offset * dim_partial;
    } else {
        // Cache hit - 更新LRU
        cache_hits++;
        lru->touch(cache_page_id);
        return cache_data + cache_page_id * page_size * dim_partial + info.offset * dim_partial;
    }
}

void PageCache::prefetch_cluster(int cluster_id, cudaStream_t stream) {
    if (cluster_id < 0 || cluster_id >= num_clusters) {
        throw std::out_of_range("prefetch_cluster: cluster_id 超出范围");
    }
    int page_count = cluster_page_count.empty() ? 0 : cluster_page_count[cluster_id];
    if (page_count == 0) {
        return;
    }

    cudaStream_t use_stream = resolve_stream(stream);
    for (int local_page_id = 0; local_page_id < page_count; local_page_id++) {
        int global_page_id = get_global_page_id(cluster_id, local_page_id);

        // 如果不在cache中，加载它
        if (cluster_to_page[global_page_id] == -1) {
            load_page(cluster_id, local_page_id, use_stream);
        } else {
            // 如果已在cache中，touch它
            lru->touch(cluster_to_page[global_page_id]);
        }
    }
}

void PageCache::prefetch_clusters_async(const std::vector<int>& cluster_ids, cudaStream_t stream) {
    cudaStream_t use_stream = resolve_stream(stream);
    for (int cluster_id : cluster_ids) {
        prefetch_cluster(cluster_id, use_stream);
    }
}

void PageCache::copy_point_to_host(int point_id, float* host_buffer, cudaStream_t stream) {
    if (!host_buffer) {
        throw std::invalid_argument("copy_point_to_host: host_buffer 不能为空");
    }
    cudaStream_t use_stream = resolve_stream(stream);
    float* device_ptr = get_point(point_id, use_stream);
    CUDA_CHECK(cudaMemcpyAsync(host_buffer, device_ptr,
                               static_cast<size_t>(dim_partial) * sizeof(float),
                               cudaMemcpyDeviceToHost, use_stream));
}

void PageCache::print_stats() const {
    long long total = cache_hits + cache_misses;
    std::cout << "=== Cache统计信息 ===" << std::endl;
    std::cout << "  Cache Hits: " << cache_hits << std::endl;
    std::cout << "  Cache Misses: " << cache_misses << std::endl;
    std::cout << "  Total Access: " << total << std::endl;
    if (total > 0) {
        std::cout << "  Hit Rate: " << (double)cache_hits / total * 100.0 << "%" << std::endl;
    }
}

float* PageCache::load_page(int cluster_id, int local_page_id, cudaStream_t stream) {
    if (cluster_page_count[cluster_id] == 0 || local_page_id >= cluster_page_count[cluster_id]) {
        throw std::out_of_range("load_page: 请求的page不存在");
    }

    int global_page_id = get_global_page_id(cluster_id, local_page_id);
    // std::cout<<"total_cluster_pages = "<<total_cluster_pages<<std::endl;
    // std::cout<<"global_page_id = "<<global_page_id<<std::endl;
    if(global_page_id >= total_cluster_pages || global_page_id < 0) {
        printf("global_page_id = %d, total_cluster_pages = %d\n", global_page_id, total_cluster_pages);
        throw std::out_of_range("load_page: 请求的page不存在");
    }

    // 使用LRU插入新page
    int old_cluster_id, old_local_page_id;
    int victim_cache_page_id = lru->insert(cluster_id, local_page_id, old_cluster_id, old_local_page_id);
    // std::cout<<"victim_cache_page_id = "<<victim_cache_page_id<<std::endl;
    // std::cout<<"old_cluster_id = "<<old_cluster_id<<std::endl;
    // std::cout<<"old_local_page_id = "<<old_local_page_id<<std::endl;

    // 清除旧映射
    if (old_cluster_id >= 0) {
        int old_global_page_id = get_global_page_id(old_cluster_id, old_local_page_id);

        cluster_to_page[old_global_page_id] = -1;
        if (device_cluster_to_page) {
            CUDA_CHECK(cudaMemcpyAsync(device_cluster_to_page + old_global_page_id,
                                       cluster_to_page.data() + old_global_page_id,
                                       sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
        }
    }

    // 建立新映射
    cluster_to_page[global_page_id] = victim_cache_page_id;
    // cudaDeviceSynchronize();
    // std::cout<<"cluster_to_page[global_page_id] = "<<cluster_to_page[global_page_id]<<std::endl;
    
    if (device_cluster_to_page) {
        CUDA_CHECK(cudaMemcpyAsync(device_cluster_to_page + global_page_id,
                                   cluster_to_page.data() + global_page_id,
                                   sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
    }

    // 从full_data复制数据到cache
    int source_start_idx = get_point_index_in_full_data(cluster_id, local_page_id, 0);
    float* cache_addr = cache_data + static_cast<size_t>(victim_cache_page_id) * page_size * dim_partial;
    const float* src_addr = full_data + static_cast<size_t>(source_start_idx) * dim_partial;

    CUDA_CHECK(cudaMemcpyAsync(cache_addr, src_addr,
                               static_cast<size_t>(page_size) * dim_partial * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    // CUDA_CHECK(cudaMemcpy(cache_addr, src_addr,
    //     static_cast<size_t>(page_size) * dim_partial * sizeof(float),
    //     cudaMemcpyHostToDevice));

    return cache_addr;
}

int PageCache::get_point_index_in_full_data(int cluster_id, int local_page_id, int offset) const {
    int cluster_base = cluster_page_offset[cluster_id] * page_size;
    int page_base = local_page_id * page_size;
    return cluster_base + page_base + offset;
}

int PageCache::get_global_page_id(int cluster_id, int local_page_id) const {
    return cluster_page_offset[cluster_id] + local_page_id;
}

cudaStream_t PageCache::resolve_stream(cudaStream_t stream) const {
    if (stream == nullptr) {
        return default_stream;
    }
    return stream;
}

