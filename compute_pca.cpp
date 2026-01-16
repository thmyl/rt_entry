#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <Eigen/Dense>
#include "pca/pca.h"
#include <sys/stat.h>
#include <sys/types.h>

// 读取fvecs格式文件
std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
    std::vector<std::vector<float>> vectors;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }
    
    while (true) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (file.eof()) break;
        
        std::vector<float> vector(dim);
        file.read(reinterpret_cast<char*>(vector.data()), dim * sizeof(float));
        if (file.eof()) break;
        
        vectors.push_back(vector);
    }
    
    file.close();
    return vectors;
}

// 读取fbin格式文件
std::vector<std::vector<float>> read_fbin(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }
    
    int n, d;
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    file.read(reinterpret_cast<char*>(&d), sizeof(int));
    
    std::vector<std::vector<float>> vectors(n, std::vector<float>(d));
    for (int i = 0; i < n; i++) {
        file.read(reinterpret_cast<char*>(vectors[i].data()), d * sizeof(float));
    }
    
    file.close();
    return vectors;
}

// 读取bvecs格式文件
std::vector<std::vector<float>> read_bvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }
    
    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));
    
    // 计算文件大小以确定向量数量
    file.seekg(0, std::ios::end);
    long long filelength = file.tellg();
    file.seekg(0, std::ios::beg);
    
    int n = filelength / (d + 4);
    printf("data shape: n = %d, d = %d\n", n, d);
    if (n > 100000000) n = 100000000; // 读取前100M
    
    std::vector<std::vector<float>> vectors;
    vectors.reserve(n);
    
    file.seekg(0, std::ios::beg);
    
    for (int i = 0; i < n; i++) {
        // 跳过每个向量前的维度标识（4字节）
        file.seekg(4, std::ios::cur);
        if (file.eof()) break;
        
        std::vector<unsigned char> tmp_data(d);
        file.read(reinterpret_cast<char*>(tmp_data.data()), d);
        if (file.eof()) break;
        
        std::vector<float> vector(d);
        for (int j = 0; j < d; j++) {
            vector[j] = static_cast<float>(tmp_data[j]);
        }
        vectors.push_back(vector);
    }
    
    file.close();
    return vectors;
}

// 根据文件后缀自动选择读取函数
std::vector<std::vector<float>> read_vectors(const std::string& filename) {
    if (filename.length() >= 5 && filename.substr(filename.length() - 5) == ".fbin") {
        return read_fbin(filename);
    } else if (filename.length() >= 6 && filename.substr(filename.length() - 6) == ".fvecs") {
        return read_fvecs(filename);
    } else if (filename.length() >= 6 && filename.substr(filename.length() - 6) == ".bvecs") {
        return read_bvecs(filename);
    } else {
        // 默认尝试使用 fvecs 格式
        std::cerr << "警告: 未知文件格式，尝试使用 fvecs 格式读取: " << filename << std::endl;
        return read_fvecs(filename);
    }
}

// 读取ivecs格式文件
std::vector<std::vector<int>> read_ivecs(const std::string& filename) {
    std::vector<std::vector<int>> vectors;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }
    
    while (true) {
        int k;
        file.read(reinterpret_cast<char*>(&k), sizeof(int));
        if (file.eof()) break;
        
        std::vector<int> vector(k);
        file.read(reinterpret_cast<char*>(vector.data()), k * sizeof(int));
        if (file.eof()) break;
        
        vectors.push_back(vector);
    }
    
    file.close();
    return vectors;
}

bool file_exists(const char* path) {
    std::ifstream file(path);
    return file.good();
}

static std::string get_filename(const std::string& path) {
    size_t pos = path.find_last_of("/");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static std::string get_dataset_name_from_path(const std::string& dataset_path) {
    std::string fname = get_filename(dataset_path);
    // e.g., sift1M_base.fvecs -> sift1M
    size_t under = fname.find_first_of("_");
    if (under == std::string::npos) return fname;
    return fname.substr(0, under);
}

static void ensure_dir(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) != 0) {
        mkdir(dir.c_str(), 0755);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <test_nq> [delta_d] [topk]" << std::endl;
        std::cerr << "  test_nq: 用于线性拟合的query数量" << std::endl;
        std::cerr << "  delta_d: 线性回归分块大小（默认 D/4）" << std::endl;
        std::cerr << "  topk: 线性拟合中使用的topk值（默认100）" << std::endl;
        return 1;
    }
    
    int test_nq = std::atoi(argv[1]);
    int delta_d_cli = -1;
    if (argc >= 3) {
        int delta_d_arg = std::atoi(argv[2]);
        if (delta_d_arg > 0) delta_d_cli = delta_d_arg;
    }
    int topk_cli = 100;  // 默认值
    if (argc >= 4) topk_cli = std::max(1, std::atoi(argv[3]));
    
    // std::string dataset_path = "/data/myl/sift1M/sift1M_base.fvecs";// TODO: change dataset path
    // std::string queryset_path = "/data/myl/sift1M/sift1M_query.fvecs";// TODO: change dataset path
    // // std::string queryset_path = "/home/myl/cache_search/gen_query/sift1M_query.fvecs";
    // std::string groundtruth_path = "/data/myl/sift1M/sift1M_groundtruth.ivecs";// TODO: change dataset path
    // // std::string groundtruth_path = "/home/myl/cache_search/gen_query/sift1M_groundtruth.ivecs";

    // std::string queryset_path = "/data/myl/deep1M/deep1M_queries.fvecs";// TODO: change dataset path
    // std::string groundtruth_path = "/data/myl/deep1M/deep1M_gt.ivecs";// TODO: change dataset path
    // std::string dataset_path = "/data/myl/deep1M/deep1M_base.fvecs";// TODO: change dataset path

    // std::string dataset_path = "/mnt/IntelP5520_8T_1/myl/sift100M/sift100M_base.fbin";// TODO: change dataset path
    // std::string queryset_path = "/mnt/IntelP5520_8T_1/myl/sift100M/sift100M_learn.fvecs";// TODO: change dataset path
    // std::string groundtruth_path = "/mnt/IntelP5520_8T_1/myl/sift100M/sift100M_learn_groundtruth.ivecs";// TODO: change dataset path

    std::string dataset_path = "/mnt/IntelP5520_8T_1/myl/deep100M/fbin/deep100M_base.fbin";// TODO: change dataset path
    std::string queryset_path = "/mnt/IntelP5520_8T_1/myl/deep100M/deep100M_learn.fvecs";// TODO: change dataset path
    std::string groundtruth_path = "/mnt/IntelP5520_8T_1/myl/deep100M/deep100M_learn_groundtruth.ivecs";// TODO: change dataset path
    
    std::string dataset_name = get_dataset_name_from_path(dataset_path);
    std::string data_root = "/mnt/IntelP5520_8T_1/myl/cache_search/data/" + dataset_name;
    // ensure_dir("data");
    ensure_dir(data_root);
    std::string log_file = data_root + "/preprocess.log";
    std::ofstream log_ofs(log_file, std::ios::app);
    auto log = [&](const std::string& s){ std::cout << s << std::endl; if (log_ofs) { log_ofs << s << std::endl; } };

    std::string meanfile = data_root + "/mean.fbin";
    std::string rotationfile = data_root + "/rotation.fbin";
    std::string rotated_base_file = data_root + "/rotated_base.fbin";
    std::string rotated_query_file = data_root + "/rotated_query.fbin";
    std::string linear_params_file = data_root + "/linear_params.bin";
    
    // 一、PCA计算
    PCA pca;
    
    // 检查是否已有PCA结果
    if (file_exists(meanfile.c_str()) && file_exists(rotationfile.c_str())) {
        log("读取已有的PCA结果...");
        pca.read_mean_rotation(meanfile.c_str(), rotationfile.c_str());
        log("PCA结果读取完成，维度: " + std::to_string(pca.dim));
    } else {
        log("开始计算PCA...");
        auto dataset = read_vectors(dataset_path);
        log("数据集大小: " + std::to_string(dataset.size()) + " x " + std::to_string(dataset[0].size()));
        
        // 准备数据
        float* dataset_data = new float[1LL * dataset.size() * dataset[0].size()];
        log("dataset_data shape: " + std::to_string(dataset.size()) + " x " + std::to_string(dataset[0].size()));
        for (int i = 0; i < dataset.size(); i++) {
            for (int j = 0; j < dataset[0].size(); j++) {
                dataset_data[1LL * i * dataset[0].size() + j] = dataset[i][j];
            }
        }
        int dataset_n = dataset.size();
        int dataset_d = dataset[0].size();
        
        // 释放dataset内存，因为数据已经复制到dataset_data
        std::vector<std::vector<float>>().swap(dataset);
        
        // 创建PCA对象并计算
        pca = PCA(dataset_data, dataset_n, dataset_d);
        pca.calc_eigenvalues();
        
        // 保存结果
        log("保存PCA结果...");
        pca.save_mean_rotation(meanfile.c_str(), rotationfile.c_str());
        
        delete[] dataset_data;
        log("PCA计算完成！");
    }
    
    // 二、线性参数计算
    if (file_exists(linear_params_file.c_str())) {
        log("读取已有的线性参数...");
        pca.read_linear_params(linear_params_file.c_str());
        log("线性参数读取完成，模型数量: " + std::to_string(pca.w.size()));
    } else {
        log("开始计算线性参数...");
        
        // 读取query和groundtruth
        auto queries = read_vectors(queryset_path);
        auto groundtruth = read_ivecs(groundtruth_path);
        
        // 限制用于训练的数据量
        int nq = std::min(test_nq, (int)queries.size());
        log("使用 " + std::to_string(nq) + " 个查询进行线性拟合");
        
        // 准备数据
        log("pca.dim = " + std::to_string(pca.dim));
        log("nq = " + std::to_string(nq));
        float* query_data = new float[1LL * nq * pca.dim];
        log("query_data shape: " + std::to_string(nq) + " x " + std::to_string(pca.dim));

        log("groundtruth shape: " + std::to_string(groundtruth.size()) + " x " + std::to_string(groundtruth[0].size()));
        int* groundtruth_data = new int[nq * groundtruth[0].size()];
        log("groundtruth_data shape: " + std::to_string(nq) + " x " + std::to_string(groundtruth[0].size()));
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < pca.dim; j++) {
                query_data[i * pca.dim + j] = queries[i][j];
            }
            for (int j = 0; j < groundtruth[0].size(); j++) {
                groundtruth_data[i * groundtruth[0].size() + j] = groundtruth[i][j];
            }
        }
        
        // 准备dataset进行PCA旋转
        auto dataset = read_vectors(dataset_path);
        float* dataset_data = new float[1LL * dataset.size() * pca.dim];
        printf("dataset_data shape: %lld x %lld\n", 1LL * dataset.size(), 1LL * pca.dim);
        
        for (int i = 0; i < dataset.size(); i++) {
            for (int j = 0; j < pca.dim; j++) {
                dataset_data[1LL * i * pca.dim + j] = dataset[i][j];
            }
        }
        
        // 保存dataset大小，然后释放dataset内存
        int dataset_size = dataset.size();
        std::vector<std::vector<float>>().swap(dataset);
        
        // 对dataset进行PCA旋转（或读取缓存）
        Eigen::MatrixXf rotated_data;
        if (file_exists(rotated_base_file.c_str())) {
            log("检测到缓存的旋转后的dataset，直接读取: " + rotated_base_file);
            std::ifstream ifs(rotated_base_file, std::ios::binary);
            uint nb_read, dim_read;
            ifs.read(reinterpret_cast<char*>(&nb_read), sizeof(uint));
            ifs.read(reinterpret_cast<char*>(&dim_read), sizeof(uint));
            rotated_data.resize(nb_read, dim_read);
            for (uint i = 0; i < nb_read; i++) {
                for (uint j = 0; j < dim_read; j++) {
                    float tmp;
                    ifs.read(reinterpret_cast<char*>(&tmp), sizeof(float));
                    rotated_data(i, j) = tmp;
                }
            }
            ifs.close();
        } else {
            log("对dataset进行PCA旋转...");
            Eigen::MatrixXf data_matrix(dataset_size, pca.dim);
            for (int i = 0; i < dataset_size; i++) {
                for (int j = 0; j < pca.dim; j++) {
                    data_matrix(i, j) = dataset_data[1LL * i * pca.dim + j] - pca.meanvecRow(j);
                }
            }
            delete[] dataset_data;
            log("data_matrix shape: " + std::to_string(data_matrix.rows()) + " x " + std::to_string(data_matrix.cols()));
            rotated_data = data_matrix * pca.vec.cast<float>();
            // 释放data_matrix内存
            Eigen::MatrixXf().swap(data_matrix);
            // 保存到缓存文件
            log("保存旋转后的dataset到文件: " + rotated_base_file);
            std::ofstream ofs(rotated_base_file, std::ios::binary);
            uint nb_u = (uint)dataset_size;
            uint dim_u = (uint)pca.dim;
            ofs.write(reinterpret_cast<const char*>(&nb_u), sizeof(uint));
            ofs.write(reinterpret_cast<const char*>(&dim_u), sizeof(uint));
            for (uint i = 0; i < nb_u; i++) {
                for (uint j = 0; j < dim_u; j++) {
                    float tmp = (float)rotated_data(i, j);
                    ofs.write(reinterpret_cast<const char*>(&tmp), sizeof(float));
                }
            }
            ofs.close();
        }
        
        // 对query进行PCA旋转（保证保存全量 queries）
        Eigen::MatrixXf rotated_query_all;
        bool need_recompute_queries = true;
        if (file_exists(rotated_query_file.c_str())) {
            std::ifstream ifs(rotated_query_file, std::ios::binary);
            uint nq_read = 0, dim_read = 0;
            ifs.read(reinterpret_cast<char*>(&nq_read), sizeof(uint));
            ifs.read(reinterpret_cast<char*>(&dim_read), sizeof(uint));
            if (nq_read == (uint)queries.size() && dim_read == (uint)pca.dim) {
                log("检测到缓存的旋转后的query（全量），直接读取: " + rotated_query_file);
                rotated_query_all.resize(nq_read, dim_read);
                for (uint i = 0; i < nq_read; i++) {
                    for (uint j = 0; j < dim_read; j++) {
                        float tmp;
                        ifs.read(reinterpret_cast<char*>(&tmp), sizeof(float));
                        rotated_query_all(i, j) = tmp;
                    }
                }
                need_recompute_queries = false;
            }
        }
        if (need_recompute_queries) {
            log("对全量query进行PCA旋转...");
            int all_nq = (int)queries.size();
            Eigen::MatrixXf query_matrix(all_nq, pca.dim);
            for (int i = 0; i < all_nq; i++) {
                for (int j = 0; j < pca.dim; j++) {
                    query_matrix(i, j) = queries[i][j] - pca.meanvecRow(j);
                }
            }
            rotated_query_all = query_matrix * pca.vec.cast<float>();
            // 保存全量到缓存文件
            log("保存全量旋转后的query到文件: " + rotated_query_file);
            std::ofstream ofs(rotated_query_file, std::ios::binary);
            uint nq_u = (uint)queries.size();
            uint dim_u = (uint)pca.dim;
            ofs.write(reinterpret_cast<const char*>(&nq_u), sizeof(uint));
            ofs.write(reinterpret_cast<const char*>(&dim_u), sizeof(uint));
            for (uint i = 0; i < nq_u; i++) {
                for (uint j = 0; j < dim_u; j++) {
                    float tmp = (float)rotated_query_all(i, j);
                    ofs.write(reinterpret_cast<const char*>(&tmp), sizeof(float));
                }
            }
            ofs.close();
        }

        // 为线性拟合准备仅 test_nq 条旋转query到数组 rotated_query_data
        Eigen::MatrixXf rotated_query = rotated_query_all.topRows(nq);
        
        // 转换为float数组
        float* rotated_dataset = new float[1LL * dataset_size * pca.dim];
        printf("rotated_dataset shape: %lld x %lld\n", 1LL * dataset_size, 1LL * pca.dim);
        float* rotated_query_data = new float[nq * pca.dim];
        
        for (int i = 0; i < dataset_size; i++) {
            for (int j = 0; j < pca.dim; j++) {
                rotated_dataset[1LL * i * pca.dim + j] = rotated_data(i, j);
            }
        }
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < pca.dim; j++) {
                rotated_query_data[i * pca.dim + j] = rotated_query(i, j);
            }
        }
        
        // 调用linear函数计算线性参数
        log("计算线性拟合参数...");
        int D = pca.dim;
        int delta_d = (delta_d_cli > 0) ? delta_d_cli : std::max(1, D / 4);
        pca.linear(rotated_dataset, rotated_query_data, groundtruth_data,
                   dataset_size, nq, nq, topk_cli, groundtruth[0].size(), D, delta_d);
        
        // 保存结果
        log("保存线性参数...");
        pca.save_linear_params(linear_params_file.c_str());
        
        delete[] query_data;
        delete[] groundtruth_data;
        // delete[] dataset_data;
        delete[] rotated_dataset;
        delete[] rotated_query_data;
        
        log("线性参数计算完成！");
    }
    
    log("所有预处理完成！");
    return 0;
}

