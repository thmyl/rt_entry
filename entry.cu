#include <iostream>
#include <cstdio>
#include "raytracing.h"
#include "entry.h"
#include "pca.h"
#include "aabb.h"

struct add_x {
    uint x;

    add_x(uint _x) : x(_x) {}

    __host__ __device__
    uint operator()(const uint& a) const {
        return a + x;
    }
};

void RT_Entry::BlockUp(){
  printf("BlockUp function\n");
  aabb_offset.resize(n_subspaces + 1);
  thrust::fill(aabb_offset.begin(), aabb_offset.end(), 0);
  // for(int space = 0; space < n_subspaces; space++){
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
    uint sub_np = subspaces_[space].sub_np;

    computeMinMax(world_bounds, h_points);
    #ifdef DETAIL
      printf("sub_np = %d\n", sub_np);
      printf("world bound of space%d: (%f, %f, %f) - (%f, %f, %f)\n", space, 
            world_bounds.minX, world_bounds.minY, world_bounds.minZ, 
            world_bounds.maxX, world_bounds.maxY, world_bounds.maxZ);
    #endif

    float3* h_points_ptr = thrust::raw_pointer_cast(subspaces_[space].h_points.data());
    // printf("n_points = %d\n", subspaces_[space].h_points.size());
    Kdtree kdtree(h_points_ptr, sub_np, sub_np * point_ratio, world_bounds, h_aabbs);
    // printf("kdtree built\n");
    n_aabbs = h_aabbs.size();
    // printf("n_aabbs = %d\n", n_aabbs);
    aabb_offset[space+1] = aabb_offset[space] + n_aabbs;
    d_aabbs.resize(n_aabbs);
    thrust::copy(h_aabbs.begin(), h_aabbs.end(), d_aabbs.begin());

    prefix_sum.resize(n_aabbs + 1);
    aabb_pid.resize(sub_np);
    kdtree.computeAabbPid(aabb_pid, prefix_sum, n_aabbs);

    expandAabb(h_aabbs, expand_ratio);
    thrust::copy(h_aabbs.begin(), h_aabbs.end(), d_aabbs.begin());
    h_aabbs.resize(0);
  }

  // conformity
  n_aabbs_ = aabb_offset[n_subspaces];
  d_prefix_sum_.resize(n_aabbs_ + 1);
  thrust::fill(d_prefix_sum_.begin(), d_prefix_sum_.end(), 0);
  d_aabb_pid_.resize(np);
  uint sum_np = 0;
  for(int space = 0; space < n_subspaces; space++){
    auto &aabb_pid = subspaces_[space].aabb_pid;
    auto &prefix_sum = subspaces_[space].prefix_sum;
    uint sub_np = subspaces_[space].sub_np;
    uint n_aabbs = subspaces_[space].n_aabbs;
    // for(int i=0; i<sub_np; i++){
    //   d_aabb_pid_[i] = aabb_pid[i] + aabb_offset[space];
    // }
    // thrust::copy(prefix_sum.begin(), prefix_sum.end(), d_prefix_sum_.begin() + aabb_offset[space]);
    thrust::copy(aabb_pid.begin(), aabb_pid.end(), d_aabb_pid_.begin() + sum_np);
    thrust::copy(prefix_sum.begin(), prefix_sum.end(), d_prefix_sum_.begin() + aabb_offset[space]);
    
    thrust::transform(d_prefix_sum_.begin() + aabb_offset[space], 
                      d_prefix_sum_.begin() + aabb_offset[space+1], 
                      d_prefix_sum_.begin() + aabb_offset[space], add_x(sum_np));
    
    sum_np += sub_np;
  }
  d_prefix_sum_[n_aabbs_] = sum_np;

  printf("n_aabbs_ = %d\n", n_aabbs_);
  printf("prefix_sum[n_aabbs_ - 1] = "); std::cout<<d_prefix_sum_[n_aabbs_ - 1]<<std::endl;
  printf("sum_np = %d\n", sum_np);
}

void RT_Entry::InitRT(){
  for(int space = 0; space < n_subspaces; space++){
    auto &rt = subspaces_[space].rt;
    auto &d_aabbs = subspaces_[space].d_aabbs;
    auto &n_aabbs = subspaces_[space].n_aabbs;
    auto *d_aabbs_ptr = thrust::raw_pointer_cast(d_aabbs.data());
    rt.Setup();
    rt.BuildAccel(d_aabbs_ptr, n_aabbs);

    // subspaces_[space].hits.resize(nq * max_hits);
    // subspaces_[space].n_hits_per_query.resize(nq);
    // subspaces_[space].hits_offset.resize(nq * max_hits);
    d_hits_.resize(nq);

    d_aabbs.resize(0);
  }
}

void RT_Entry::Search(thrust::device_vector<float> &d_pca_queries, thrust::device_vector<uint> &d_gt_, thrust::device_vector<uint> &d_entries, thrust::device_vector<float> &d_entries_dist, uint n_entries){
  // Timing::startTiming("search_entry");
  //rt搜索
  #ifdef DETAIL
    Timing::startTiming("rt search");
  #endif
  for(uint space = 0; space < n_subspaces; space++){
    auto &rt = subspaces_[space].rt;
    auto &hits = d_hits_;
    // auto &n_hits_per_query = subspaces_[space].n_hits_per_query;
    printf("dim_ = %d\n", dim_);
    rt.search(thrust::raw_pointer_cast(d_pca_queries.data()), nq, space*3, dim_, thrust::raw_pointer_cast(hits.data()));
    
    /*thrust::host_vector<uint> h_n_hits_per_query = n_hits_per_query;
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

void RT_Entry::Search(KPCA* kpca, thrust::device_vector<uint> &d_belong, thrust::device_vector<uint> &d_gt_, thrust::device_vector<uint> &d_entries, thrust::device_vector<float> &d_entries_dist, uint n_entries){
  // Timing::startTiming("search_entry");
  //rt搜索
  #ifdef DETAIL
    Timing::startTiming("rt search");
  #endif
  for(uint space = 0; space < n_subspaces; space++){
    auto &rt = subspaces_[space].rt;
    auto &hits = d_hits_;
    auto &d_pca_queries = kpca->d_transforms[space];
    uint offset = aabb_offset[space];
    #ifdef DETAIL
    printf("dim_ = %d\n", dim_);
    #endif

    uint* d_belong_ptr = thrust::raw_pointer_cast(d_belong.data());
    rt.search(thrust::raw_pointer_cast(d_pca_queries.data()), nq, offset, dim_, thrust::raw_pointer_cast(hits.data()), d_belong_ptr, space);
  }
  #ifdef DETAIL
    Timing::stopTiming(2);
  #endif
}

void RT_Entry::CleanUp(){
  for(int space = 0; space < n_subspaces; space++){
    auto &rt = subspaces_[space].rt;
    rt.CleanUp();
  }
}

void RT_Entry::set_pca_points(thrust::host_vector<float> &h_pca_points, uint points_dim){
  long long sizeof_points = h_pca_points.size();
  std::cout<<"sizeof_points = "<<sizeof_points<<std::endl;
  for(int space = 0; space < n_subspaces; space++){
    subspaces_[space].sub_np = np;
    subspaces_[space].h_points.resize(np);
    for(long long i=0; i<np; i++){
      long long offset = i*points_dim + space*3;
      subspaces_[space].h_points[i] = float3{h_pca_points[offset], h_pca_points[offset+1], h_pca_points[offset+2]};
    }
  }
}

void RT_Entry::set_pca_clusters(KPCA* kpca){
  printf("set_pca_clusters\n");
  uint n_components = kpca->n_components;
  for(int space = 0; space < n_subspaces; space++){
    uint _n = kpca->h_cluster_lists[space].size();
    // printf("_n = %d\n", _n);
    subspaces_[space].sub_np = _n;
    subspaces_[space].h_points.resize(_n);

    for(long long i=0; i<_n; i++){
      long long offset = i*n_components;
      subspaces_[space].h_points[i] = float3{
        kpca->h_pca_bases[space][offset],
        kpca->h_pca_bases[space][offset+1],
        kpca->h_pca_bases[space][offset+2]
      };
    }
  }
}
