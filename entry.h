#pragma once

#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cublas_v2.h>
#include "head.h"
#include "raytracing.h"
#include "helper.h"
#include "file_read.h"
#include "matrix.h"

class Entry{
  public:
    Entry(){
      rotation_matrix_path = data_name + "/O.fvecs";
      mean_matrix_path = data_name + "/mean.fvecs";
      pca_base_path = data_name + "/pca_base.fvecs";
      // pca_query_path = data_name + "/pca_query.fvecs";
      n_subspaces = 1;
      buffer_size = 200000;
      // buffer_size = 10000;
      // buffer_size = 12288;
      cublasStatus_t status = cublasCreate(&handle_);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "!!!! CUBLAS initialization error\n";
        return;
      }
    }
    Entry(uint n_subspaces_, uint buffer_size_, std::string data_name_, uint entries_size_, uint max_hits_, double expand_ratio_, double point_ratio_, double grid_size_){
      n_subspaces = n_subspaces_;
      buffer_size = buffer_size_;
      data_name = data_name_;
      rotation_matrix_path = data_name + "/O.fvecs";
      mean_matrix_path = data_name + "/mean.fvecs";
      pca_base_path = data_name + "/pca_base.fvecs";
      entries_size = entries_size_;
      max_hits = max_hits_;
      expand_ratio = static_cast<float>(expand_ratio_);
      point_ratio = static_cast<float>(point_ratio_);
      grid_size = static_cast<float>(grid_size_);

      cublasStatus_t status = cublasCreate(&handle_);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "!!!! CUBLAS initialization error\n";
        return;
      }
    }
    void Input(char*, char*, char*);
    void Projection();
    void BlockUp();
    void InitRT();
    void CleanUp();
    void Search();
    void collect_candidates_onesubspace(thrust::device_vector<uint> &hits,
                                        thrust::device_vector<uint> &n_hits_per_query,
                                        uint &max_hits, 
                                        thrust::device_vector<uint> &aabb_pid,
                                        thrust::device_vector<uint> &prefix_sum,
                                        uint &n_aabbs);
    void check_candidates();
    void check_entries();
    void subspace_copy(thrust::device_vector<float3> &d_dst, thrust::device_vector<float> &d_src, uint offset);
  
  private:
    std::string          data_name = "sift1M";
    std::string          data_path;
    std::string          query_path;
    std::string          gt_path;
    std::string          rotation_matrix_path;
    std::string          mean_matrix_path;
    std::string          pca_base_path;
    // std::string          pca_query_path;
    thrust::host_vector<float> h_points_;
    thrust::host_vector<float> h_queries_;
    thrust::host_vector<uint> h_gt_;
    thrust::device_vector<float> d_points_;
    thrust::device_vector<float> d_queries_;
    thrust::device_vector<uint> d_gt_;
    thrust::host_vector<float> h_mean;
    thrust::device_vector<float> d_mean;
    thrust::host_vector<float> h_rotation;
    thrust::device_vector<float> d_rotation;
    // thrust::host_vector<float> h_mr;
    // thrust::device_vector<float> d_mr;
    thrust::device_vector<float> d_mr_row;
    // thrust::host_vector<float> h_pca_queries;
    thrust::device_vector<float> d_pca_queries;
    thrust::host_vector<float> h_pca_points;
    thrust::device_vector<float> d_pca_points;
    uint                 dim_;
    uint                 nq;
    uint                 np;
    uint                 gt_k;
    thrust::host_vector<uint> h_candidates;
    thrust::device_vector<uint> d_candidates;
    thrust::device_vector<float> d_candidates_dist;
    thrust::host_vector<uint> h_n_candidates;
    thrust::device_vector<uint> d_n_candidates;
    thrust::device_vector<uint> d_entries;
    thrust::device_vector<float> d_entries_dist;
    cublasHandle_t       handle_;
    uint                 entries_size = 64;
    uint                 buffer_size;

    uint max_hits = 80;
    float expand_ratio = 0.8f;
    float point_ratio = 0.0025f;
    float grid_size = 16.0f;

  private:
    uint n_subspaces = 1;
    struct Subspace{
      thrust::host_vector<float3> h_points;
      // thrust::host_vector<float3> h_queries;
      thrust::device_vector<float3> d_points;
      // thrust::device_vector<float3> d_queries;
      // thrust::host_vector<float> h_rotation;
      // thrust::device_vector<float> d_rotation;

      OptixAabb world_bounds;
      thrust::host_vector<OptixAabb> h_aabbs;
      thrust::device_vector<OptixAabb> d_aabbs;
      thrust::device_vector<uint> prefix_sum;
      thrust::device_vector<uint> aabb_pid;
      uint n_aabbs;
      
      thrust::device_vector<uint> hits;
      thrust::device_vector<uint> n_hits_per_query;

      OptiXRT rt;
    };
    std::vector<Subspace> subspaces_;
};