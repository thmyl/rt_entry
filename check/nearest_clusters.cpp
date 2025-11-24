#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct QueryDataset {
    int dim = 0;
    std::vector<float> data;
};

QueryDataset read_fvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("无法打开 query 文件: " + path);
    }

    QueryDataset result;
    std::vector<float> buffer;
    buffer.reserve(1024);

    while (true) {
        int dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!in) {
            if (in.eof()) {
                break;
            }
            throw std::runtime_error("读取 query 文件失败: " + path);
        }

        if (result.dim == 0) {
            result.dim = dim;
        } else if (dim != result.dim) {
            throw std::runtime_error("query 文件中存在不同的维度记录");
        }

        std::vector<float> vec(static_cast<size_t>(dim));
        in.read(reinterpret_cast<char*>(vec.data()), sizeof(float) * vec.size());
        if (!in) {
            throw std::runtime_error("读取 query 向量数据失败");
        }

        result.data.insert(result.data.end(), vec.begin(), vec.end());
    }

    if (result.dim == 0) {
        throw std::runtime_error("query 文件为空或格式不正确");
    }

    return result;
}

struct CentroidDataset {
    int k = 0;
    int dim = 0;
    std::vector<float> centroids;
};

CentroidDataset read_centroids(const std::string& path, int dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("无法打开 centroids 文件: " + path);
    }

    in.seekg(0, std::ios::end);
    const std::streamoff total_size = in.tellg();
    in.seekg(0, std::ios::beg);

    int k = 0;
    in.read(reinterpret_cast<char*>(&k), sizeof(int));
    if (!in) {
        throw std::runtime_error("读取 centroids 文件失败: " + path);
    }

    // 文件格式:
    // int K
    // int labels[N]
    // float centroids[K][dim]
    // for each cluster: int cluster_size, int point_ids[cluster_size]
    const std::int64_t rest_bytes = static_cast<std::int64_t>(total_size) - sizeof(int);
    const std::int64_t centroid_bytes = static_cast<std::int64_t>(k) * dim * sizeof(float);
    const std::int64_t header_bytes = static_cast<std::int64_t>(k) * sizeof(int);

    const std::int64_t numerator = rest_bytes - centroid_bytes - header_bytes;
    if (numerator < 0 || numerator % (2 * static_cast<std::int64_t>(sizeof(int))) != 0) {
        throw std::runtime_error("centroids 文件大小与预期格式不一致");
    }
    const std::int64_t num_points = numerator / (2 * static_cast<std::int64_t>(sizeof(int)));

    // 跳过 labels
    in.seekg(num_points * sizeof(int), std::ios::cur);

    std::vector<float> centroids(static_cast<size_t>(k) * dim);
    in.read(reinterpret_cast<char*>(centroids.data()),
            static_cast<std::streamsize>(centroids.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error("读取 centroid 数据失败");
    }

    return CentroidDataset{k, dim, std::move(centroids)};
}

float squared_l2_distance(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

} // namespace

int main() {
    try {
        const std::string centroids_path = "/home/myl/cache_search/data/sift1M/centroids_100";
        const std::string query_path = "/data/myl/sift1M/sift1M_query.fvecs";

        std::cout << "读取 query 文件: " << query_path << std::endl;
        QueryDataset queries = read_fvecs(query_path);
        const int dim = queries.dim;
        const std::size_t num_queries = queries.data.size() / dim;
        std::cout << "query 数量: " << num_queries << ", 维度: " << dim << std::endl;

        std::cout << "读取 centroids 文件: " << centroids_path << std::endl;
        CentroidDataset centroids = read_centroids(centroids_path, dim);
        std::cout << "cluster 数量: " << centroids.k << std::endl;

        std::vector<std::pair<float, int>> distances;
        distances.reserve(static_cast<std::size_t>(centroids.k));

        for (std::size_t qi = 0; qi < 100; ++qi) {
            const float* query_vec = queries.data.data() + qi * dim;
            distances.clear();

            for (int cid = 0; cid < centroids.k; ++cid) {
                const float* centroid_vec = centroids.centroids.data() + static_cast<std::size_t>(cid) * dim;
                const float dist = squared_l2_distance(query_vec, centroid_vec, dim);
                distances.emplace_back(dist, cid);
            }

            const std::size_t top_k = std::min<std::size_t>(5, distances.size());
            std::partial_sort(distances.begin(), distances.begin() + top_k, distances.end());

            std::cout << "Query " << qi << ":";
            for (std::size_t i = 0; i < top_k; ++i) {
                std::cout << " (cid=" << distances[i].second
                          << ", dist=" << distances[i].first << ")";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "发生错误: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

