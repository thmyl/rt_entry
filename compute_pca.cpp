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
    
    std::string dataset_path = "/data/myl/sift1M/sift1M_base.fvecs";// TODO: change dataset path
    std::string queryset_path = "/data/myl/sift1M/sift1M_query.fvecs";// TODO: change dataset path
    std::string groundtruth_path = "/data/myl/sift1M/sift1M_groundtruth.ivecs";// TODO: change dataset path
    // std::string queryset_path = "/data/myl/deep1M/deep1M_queries.fvecs";// TODO: change dataset path
    // std::string groundtruth_path = "/data/myl/deep1M/deep1M_gt.ivecs";// TODO: change dataset path
    // std::string dataset_path = "/data/myl/deep1M/deep1M_base.fvecs";// TODO: change dataset path
    
    std::string dataset_name = get_dataset_name_from_path(dataset_path);
    std::string data_root = "data/" + dataset_name;
    ensure_dir("data");
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
        auto dataset = read_fvecs(dataset_path);
        log("数据集大小: " + std::to_string(dataset.size()) + " x " + std::to_string(dataset[0].size()));
        
        // 准备数据
        float* dataset_data = new float[dataset.size() * dataset[0].size()];
        for (int i = 0; i < dataset.size(); i++) {
            for (int j = 0; j < dataset[0].size(); j++) {
                dataset_data[i * dataset[0].size() + j] = dataset[i][j];
            }
        }
        
        // 创建PCA对象并计算
        pca = PCA(dataset_data, dataset.size(), dataset[0].size());
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
        auto queries = read_fvecs(queryset_path);
        auto groundtruth = read_ivecs(groundtruth_path);
        
        // 限制用于训练的数据量
        int nq = std::min(test_nq, (int)queries.size());
        log("使用 " + std::to_string(nq) + " 个查询进行线性拟合");
        
        // 准备数据
        float* query_data = new float[nq * pca.dim];
        int* groundtruth_data = new int[nq * groundtruth[0].size()];
        
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < pca.dim; j++) {
                query_data[i * pca.dim + j] = queries[i][j];
            }
            for (int j = 0; j < groundtruth[0].size(); j++) {
                groundtruth_data[i * groundtruth[0].size() + j] = groundtruth[i][j];
            }
        }
        
        // 准备dataset进行PCA旋转
        auto dataset = read_fvecs(dataset_path);
        float* dataset_data = new float[dataset.size() * pca.dim];
        
        for (int i = 0; i < dataset.size(); i++) {
            for (int j = 0; j < pca.dim; j++) {
                dataset_data[i * pca.dim + j] = dataset[i][j];
            }
        }
        
        // 对dataset进行PCA旋转（或读取缓存）
        Eigen::MatrixXd rotated_data;
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
            Eigen::MatrixXd data_matrix(dataset.size(), pca.dim);
            for (int i = 0; i < dataset.size(); i++) {
                for (int j = 0; j < pca.dim; j++) {
                    data_matrix(i, j) = dataset_data[i * pca.dim + j] - pca.meanvecRow(j);
                }
            }
            rotated_data = data_matrix * pca.vec;
            // 保存到缓存文件
            log("保存旋转后的dataset到文件: " + rotated_base_file);
            std::ofstream ofs(rotated_base_file, std::ios::binary);
            uint nb_u = (uint)dataset.size();
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
        Eigen::MatrixXd rotated_query_all;
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
            Eigen::MatrixXd query_matrix(all_nq, pca.dim);
            for (int i = 0; i < all_nq; i++) {
                for (int j = 0; j < pca.dim; j++) {
                    query_matrix(i, j) = queries[i][j] - pca.meanvecRow(j);
                }
            }
            rotated_query_all = query_matrix * pca.vec;
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
        Eigen::MatrixXd rotated_query = rotated_query_all.topRows(nq);
        
        // 转换为float数组
        float* rotated_dataset = new float[dataset.size() * pca.dim];
        float* rotated_query_data = new float[nq * pca.dim];
        
        for (int i = 0; i < dataset.size(); i++) {
            for (int j = 0; j < pca.dim; j++) {
                rotated_dataset[i * pca.dim + j] = rotated_data(i, j);
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
                   dataset.size(), nq, nq, topk_cli, groundtruth[0].size(), D, delta_d);
        
        // 保存结果
        log("保存线性参数...");
        pca.save_linear_params(linear_params_file.c_str());
        
        delete[] query_data;
        delete[] groundtruth_data;
        delete[] dataset_data;
        delete[] rotated_dataset;
        delete[] rotated_query_data;
        
        log("线性参数计算完成！");
    }
    
    log("所有预处理完成！");
    return 0;
}

