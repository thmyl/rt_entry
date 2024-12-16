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
  uint n_hits;
  uint aabb_id;
  uint st;
  uint ed;
  uint offset;
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
    RT_Entry(uint n_subspaces_, uint buffer_size_, uint max_hits_, double expand_ratio_, double point_ratio_){
      n_subspaces = n_subspaces_;
      buffer_size = buffer_size_;
      max_hits = max_hits_;
      expand_ratio = static_cast<float>(expand_ratio_);
      point_ratio = static_cast<float>(point_ratio_);

      subspaces_.resize(n_subspaces);
    }
    void BlockUp();
    void InitRT();
    void CleanUp();
    void Search(thrust::device_vector<float> &d_pca_points, thrust::device_vector<float> &d_pca_queries, thrust::device_vector<uint> &d_gt_, thrust::device_vector<uint> &d_entries, thrust::device_vector<float> &d_entries_dist, uint n_entries);
    void collect_candidates_onesubspace(
                                        thrust::device_vector<float> &d_pca_points,
                                        thrust::device_vector<float> &d_pca_queries,
                                        thrust::device_vector<uint> &d_entries,
                                        thrust::device_vector<float> &d_entries_dist,
                                        uint n_entries,
                                        thrust::device_vector<uint> &hits,
                                        thrust::device_vector<uint> &n_hits_per_query,
                                        thrust::device_vector<uint> &hits_offset,
                                        uint &max_hits, 
                                        thrust::device_vector<uint> &aabb_pid,
                                        thrust::device_vector<uint> &prefix_sum,
                                        uint &n_aabbs);
    void check_candidates(thrust::device_vector<uint> &d_gt_);
    void subspace_copy(thrust::device_vector<float3> &d_dst, thrust::device_vector<float> &d_src, uint offset);
    void set_size(uint _dim_, uint _np, uint _nq, uint _gt_k){
      dim_ = _dim_;
      np = _np;
      nq = _nq;
      gt_k = _gt_k;
    }
    uint get_n_subspaces(){return n_subspaces;}
    void set_pca_points(thrust::host_vector<float> &h_pca_points){
      for(int space = 0; space < n_subspaces; space++){
        subspaces_[space].h_points.resize(np);
        for(int i=0; i<np; i++){
          uint offset = i*dim_ + space*3;
          subspaces_[space].h_points[i] = float3{h_pca_points[offset], h_pca_points[offset+1], h_pca_points[offset+2]};
        }
      }
    }
  
  private:
    uint                 dim_;
    uint                 nq;
    uint                 np;
    uint                 gt_k;
    
    thrust::device_vector<uint> d_candidates;
    thrust::device_vector<float> d_candidates_dist;
    thrust::device_vector<uint> d_n_candidates;
    uint                 buffer_size;

    uint max_hits = 80;
    float expand_ratio = 0.8f;
    float point_ratio = 0.0025f;

  public:
    uint n_subspaces = 1;
    struct Subspace{
      thrust::host_vector<float3> h_points;

      OptixAabb world_bounds;
      thrust::host_vector<OptixAabb> h_aabbs;
      thrust::device_vector<OptixAabb> d_aabbs;
      thrust::device_vector<uint> prefix_sum;
      thrust::device_vector<uint> aabb_pid;
      uint n_aabbs;
      
      thrust::device_vector<uint> hits;
      thrust::device_vector<uint> n_hits_per_query;
      thrust::device_vector<uint> hits_offset;

      OptiXRT rt;
    };
    std::vector<Subspace> subspaces_;
};