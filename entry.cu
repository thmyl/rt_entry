#include <iostream>
#include <cstdio>
#include "raytracing.h"
#include "entry.h"
#include "pca.h"
#include "aabb.h"

void RT_Entry::BlockUp(){
  printf("BlockUp function\n");
  for(int space = 0; space < n_subspaces; space++){
    //均匀分配块
    auto &world_bounds = subspaces_[space].world_bounds;
    auto &h_points = subspaces_[space].h_points;
    // auto &d_points = subspaces_[space].d_points;
    auto &h_aabbs = subspaces_[space].h_aabbs;
    auto &d_aabbs = subspaces_[space].d_aabbs;
    auto &n_aabbs = subspaces_[space].n_aabbs;
    auto &aabb_pid = subspaces_[space].aabb_pid;
    auto &prefix_sum = subspaces_[space].prefix_sum;

    computeMinMax(world_bounds, h_points);
    #ifdef DETAIL
      printf("world bound of space%d: (%f, %f, %f) - (%f, %f, %f)\n", space, 
            world_bounds.minX, world_bounds.minY, world_bounds.minZ, 
            world_bounds.maxX, world_bounds.maxY, world_bounds.maxZ);
    #endif

    float3* h_points_ptr = thrust::raw_pointer_cast(subspaces_[space].h_points.data());
    Kdtree kdtree(h_points_ptr, np, np * point_ratio, world_bounds, h_aabbs);
    n_aabbs = h_aabbs.size();
    printf("n_aabbs = %d\n", n_aabbs);
    d_aabbs.resize(n_aabbs);
    thrust::copy(h_aabbs.begin(), h_aabbs.end(), d_aabbs.begin());

    prefix_sum.resize(n_aabbs + 1);
    aabb_pid.resize(np);
    kdtree.computeAabbPid(aabb_pid, prefix_sum, n_aabbs);

    expandAabb(h_aabbs, expand_ratio);
    thrust::copy(h_aabbs.begin(), h_aabbs.end(), d_aabbs.begin());
    h_aabbs.resize(0);
  }
}

void RT_Entry::InitRT(){
  for(int space = 0; space < n_subspaces; space++){
    auto &rt = subspaces_[space].rt;
    auto &d_aabbs = subspaces_[space].d_aabbs;
    auto &n_aabbs = subspaces_[space].n_aabbs;
    auto *d_aabbs_ptr = thrust::raw_pointer_cast(d_aabbs.data());
    rt.Setup();
    rt.BuildAccel(d_aabbs_ptr, n_aabbs);

    subspaces_[space].hits.resize(nq * max_hits);
    subspaces_[space].n_hits_per_query.resize(nq);
    subspaces_[space].hits_offset.resize(nq * max_hits);

    d_aabbs.resize(0);
  }
  // preheat_cublas(nq, dim_, dim_);
  // d_candidates.resize(nq * buffer_size);
  // d_candidates_dist.resize(nq * buffer_size);
  // d_n_candidates.resize(nq);
}

void RT_Entry::Search(thrust::device_vector<float> &d_pca_points, thrust::device_vector<float> &d_pca_queries, thrust::device_vector<int> &d_gt_, thrust::device_vector<int> &d_entries, thrust::device_vector<float> &d_entries_dist, int n_entries){
  // Timing::startTiming("search_entry");
  //rt搜索
  #ifdef DETAIL
    Timing::startTiming("rt search");
  #endif
  for(int space = 0; space < n_subspaces; space++){
    auto &rt = subspaces_[space].rt;
    auto &hits = subspaces_[space].hits;
    auto &n_hits_per_query = subspaces_[space].n_hits_per_query;
    // printf("dim_ = %d\n", dim_);
    rt.search(thrust::raw_pointer_cast(d_pca_queries.data()), nq, space*3, dim_, thrust::raw_pointer_cast(hits.data()), thrust::raw_pointer_cast(n_hits_per_query.data()), max_hits);
    
    /*thrust::host_vector<int> h_n_hits_per_query = n_hits_per_query;
    float sum = 0;
    for(int i=0; i<nq; i++){
      sum += h_n_hits_per_query[i];
      if(h_n_hits_per_query[i] == 0){
        printf("query %d has no hit\n", i);
      }
    }
    sum = sum/nq;
    printf("average hits = %f AABBs\n", sum);*/
  }
  #ifdef DETAIL
    Timing::stopTiming(2);
  #endif

  //合并搜索结果
  // Timing::startTiming("collect candidates");
  // if(n_subspaces == 1){
  //   // Timing::startTiming("before collect candidates onesubspace");
  //   auto &hits = subspaces_[0].hits;
  //   auto &n_hits_per_query = subspaces_[0].n_hits_per_query;
  //   auto &hits_offset = subspaces_[0].hits_offset;
  //   auto &aabb_pid = subspaces_[0].aabb_pid;
  //   auto &prefix_sum = subspaces_[0].prefix_sum;
  //   auto &n_aabbs = subspaces_[0].n_aabbs;
  //   // Timing::stopTiming();

  //   // Timing::startTiming("collect candidates onesubspace");
  //   // collect_candidates_onesubspace(d_pca_points, d_pca_queries, d_entries, d_entries_dist, n_entries, hits, n_hits_per_query, hits_offset, max_hits, aabb_pid, prefix_sum, n_aabbs);
  //   // Timing::stopTiming();
  // }
  // Timing::stopTiming(2);
  // Timing::stopTiming(2);
  // #ifdef DETAIL
    // check_candidates(d_gt_);
  // #endif
  // check_entries(d_gt_);
}

void RT_Entry::CleanUp(){
  for(int space = 0; space < n_subspaces; space++){
    auto &rt = subspaces_[space].rt;
    rt.CleanUp();
  }
}

void RT_Entry::set_pca_points(thrust::host_vector<float> &h_pca_points, int points_dim){
  long long sizeof_points = h_pca_points.size();
  std::cout<<"sizeof_points = "<<sizeof_points<<std::endl;
  for(int space = 0; space < n_subspaces; space++){
    subspaces_[space].h_points.resize(np);
    for(long long i=0; i<np; i++){
      long long offset = i*points_dim + space*3;
      subspaces_[space].h_points[i] = float3{h_pca_points[offset], h_pca_points[offset+1], h_pca_points[offset+2]};
    }
  }
}
