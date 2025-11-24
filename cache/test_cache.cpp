#include "page_cache.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#define CUDA_CHECK(expr)                                                                    \
    do {                                                                                    \
        cudaError_t _err = (expr);                                                          \
        if (_err != cudaSuccess) {                                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(_err) << " at "              \
                      << __FILE__ << ":" << __LINE__ << std::endl;                        \
            std::exit(EXIT_FAILURE);                                                        \
        }                                                                                   \
    } while (0)

namespace {

void print_vector(const std::vector<float>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i + 1 < vec.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
}

} // namespace

void simulate_query_batches() {
    std::cout << "\n=== 测试: GPU PageCache Query Batch 模拟 ===" << std::endl;

    const int page_size = 4;
    const int num_pages = 4;      // GPU cache 只能容纳4个page
    const int dim_partial = 3;
    const int num_clusters = 4;
    const int points_per_cluster = 8;
    const int num_points = num_clusters * points_per_cluster;

    // 构造测试数据
    float* data = new float[num_points * dim_partial];
    std::vector<int> cluster_labels(num_points);
    std::vector<std::vector<int>> cluster_points(num_clusters);

    for (int pid = 0; pid < num_points; ++pid) {
        int cluster_id = pid / points_per_cluster;
        cluster_labels[pid] = cluster_id;
        cluster_points[cluster_id].push_back(pid);
        for (int d = 0; d < dim_partial; ++d) {
            data[pid * dim_partial + d] = static_cast<float>(pid * 0.1f + d);
        }
    }

    PageCache cache(page_size, num_pages, dim_partial, num_clusters, num_points);
    cache.init_data(data, dim_partial, 0, cluster_labels, cluster_points);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<std::vector<int>> query_batches = {
        {0, 1},
        {2, 3},
        {1, 3},
        {0, 2}
    };

    std::vector<float> host_buffer(dim_partial, 0.0f);

    for (size_t batch_idx = 0; batch_idx < query_batches.size(); ++batch_idx) {
        const auto& batch = query_batches[batch_idx];
        std::cout << "\n--- Query Batch " << batch_idx << " ---" << std::endl;
        std::cout << "预加载clusters: ";
        for (int cid : batch) {
            std::cout << cid << " ";
        }
        std::cout << std::endl;

        cache.prefetch_clusters_async(batch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cache.reset_stats();
        for (int cid : batch) {
            if (cluster_points[cid].empty()) {
                continue;
            }
            int pid = cluster_points[cid].front();
            cache.copy_point_to_host(pid, host_buffer.data(), stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::cout << "  Cluster " << cid << " 首个点 (id=" << pid << "): ";
            print_vector(host_buffer);
            std::cout << std::endl;
        }
        cache.print_stats();
    }

    std::cout << "\n--- 直接点访问 (不预加载) ---" << std::endl;
    cache.reset_stats();
    std::vector<int> direct_access = {
        cluster_points[0][1],
        cluster_points[2][3],
        cluster_points[1][5],
        cluster_points[0][2]
    };

    for (int pid : direct_access) {
        cache.copy_point_to_host(pid, host_buffer.data(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cout << "  点 id=" << pid << ": ";
        print_vector(host_buffer);
        std::cout << std::endl;
    }
    cache.print_stats();

    CUDA_CHECK(cudaStreamDestroy(stream));
    delete[] data;
}

int main() {
    srand(static_cast<unsigned>(time(nullptr)));

    std::cout << "=====================================" << std::endl;
    std::cout << "   GPU Page Cache 系统测试" << std::endl;
    std::cout << "=====================================" << std::endl;

    simulate_query_batches();

    std::cout << "\n=====================================" << std::endl;
    std::cout << "   所有测试完成" << std::endl;
    std::cout << "=====================================" << std::endl;

    return 0;
}

