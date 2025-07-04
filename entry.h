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

struct QueryAabb{
  int n_hits;
  int aabb_id;
  int st;
  int ed;
  int offset;
};

class RT_Entry{
  public:
    RT_Entry(){
      // rotation_matrix_path = data_name + "/O.fvecs";
      // mean_matrix_path = data_name + "/mean.fvecs";
      // pca_base_path = data_name + "/pca_base.fvecs";
      n_subspaces = 1;
      buffer_size = 200000;
      
    }
    RT_Entry(std::string _data_name, int n_subspaces_, int buffer_size_, int max_hits_, double expand_ratio_, double point_ratio_, int n_candidates_){
      n_subspaces = n_subspaces_;
      buffer_size = buffer_size_;
      max_hits = max_hits_;
      expand_ratio = static_cast<float>(expand_ratio_);
      point_ratio = static_cast<float>(point_ratio_);
      n_candidates = n_candidates_;
      data_name = _data_name;

      subspaces_.resize(n_subspaces);
    }
    void BlockUp();
    void InitRT();
    void CleanUp();
    void Search(thrust::device_vector<float> &d_pca_points, thrust::device_vector<float> &d_pca_queries, thrust::device_vector<int> &d_gt_, thrust::device_vector<int> &d_entries, thrust::device_vector<float> &d_entries_dist, int n_entries);
    void check_candidates(thrust::device_vector<int> &d_gt_);
    void subspace_copy(thrust::device_vector<float3> &d_dst, thrust::device_vector<float> &d_src, int offset);
    void set_size(int _dim_, int _np, int _nq, int _gt_k){
      dim_ = _dim_;
      np = _np;
      nq = _nq;
      gt_k = _gt_k;
    }
    int get_n_subspaces(){return n_subspaces;}
    void set_pca_points(thrust::host_vector<float> &h_pca_points, int points_dim);
    void read_aabbs(const char* filename, int &n_aabbs, thrust::host_vector<OptixAabb>& h_aabbs, int &aabb_size, thrust::host_vector<int>& h_aabb_pid);
    void write_aabbs(const char* filename, int n_aabbs, const thrust::host_vector<OptixAabb>& h_aabbs, int aabb_size, const thrust::host_vector<int>& h_aabb_pid);
  
  private:
    int                 dim_;
    int                 nq;
    int                 np;
    int                 gt_k;
    std::string         data_name;
    
    thrust::device_vector<int> d_candidates;
    thrust::device_vector<float> d_candidates_dist;
    thrust::device_vector<int> d_n_candidates;
    int                 buffer_size;

    int max_hits = 80;
    float expand_ratio = 0.8f;
    float point_ratio = 0.0025f;
    int n_candidates;

  public:
    int n_subspaces = 1;
    struct Subspace{
      thrust::host_vector<float3> h_points;

      OptixAabb world_bounds;
      thrust::host_vector<OptixAabb> h_aabbs;
      thrust::device_vector<OptixAabb> d_aabbs;
      // thrust::device_vector<int> prefix_sum;
      // thrust::device_vector<int> aabb_pid;
      thrust::device_vector<int> aabb_entries;
      thrust::host_vector<int> h_aabb_entries;
      thrust::host_vector<int> h_aabb_pid;
      int n_aabbs;
      int aabb_size;
      
      thrust::device_vector<int> hits;

      OptiXRT rt;
    };
    std::vector<Subspace> subspaces_;
};