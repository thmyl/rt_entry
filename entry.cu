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
    auto &aabb_size = subspaces_[space].aabb_size;
    auto &aabb_entries = subspaces_[space].aabb_entries;
    auto &h_aabb_entries = subspaces_[space].h_aabb_entries;
    auto &h_aabb_pid = subspaces_[space].h_aabb_pid;
    aabb_size = np * point_ratio;
    #ifdef DETAIL
      printf("aabb_size = %d\n", aabb_size);
    #endif

    std::string aabb_file = data_name + "/" + std::to_string(aabb_size) + "_" + std::to_string(space) + ".aabbs";

    FILE* fp = fopen(aabb_file.c_str(), "rb");
    if(fp == NULL){
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

      kdtree.computeAabbPid(h_aabb_pid, n_aabbs);

      printf("writting AABB to file %s\n", aabb_file.c_str());
      write_aabbs(aabb_file.c_str(), n_aabbs, h_aabbs, aabb_size, h_aabb_pid);
    }
    else{
      fclose(fp);
      printf("reading AABB from file %s\n", aabb_file.c_str());
      read_aabbs(aabb_file.c_str(), n_aabbs, h_aabbs, aabb_size, h_aabb_pid);
    }

    //写入文件
    std::ofstream outfile;
    outfile.open(OUTFILE, std::ios_base::app);
    outfile <<  "n_aabbs: " << n_aabbs << "\n" << std::flush;
    outfile.close();

    h_aabb_entries.resize(n_aabbs * n_candidates);
    aabb_entries.resize(n_aabbs * n_candidates);
    for(int i=0; i<n_aabbs; i++){
      for(int j=0; j<n_candidates; j++){
        if(j < aabb_size)
          h_aabb_entries[i*n_candidates+j] = h_aabb_pid[i*aabb_size+j];
        else h_aabb_entries[i*n_candidates+j] = np;
      }
    }
    thrust::copy(h_aabb_entries.begin(), h_aabb_entries.end(), aabb_entries.begin());

    expandAabb(h_aabbs, expand_ratio);
    d_aabbs.resize(n_aabbs * n_candidates);
    thrust::copy(h_aabbs.begin(), h_aabbs.end(), d_aabbs.begin());
    h_aabbs.resize(0);
    h_aabb_entries.resize(0);
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
    thrust::fill(subspaces_[space].hits.begin(), subspaces_[space].hits.end(), 0);

    d_aabbs.resize(0);
    d_aabbs.shrink_to_fit();
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
    // printf("dim_ = %d\n", dim_);
    rt.search(thrust::raw_pointer_cast(d_pca_queries.data()), nq, space*3, dim_, thrust::raw_pointer_cast(hits.data()));
    
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


void RT_Entry::write_aabbs(const char* filename, int n_aabbs, const thrust::host_vector<OptixAabb>& h_aabbs, int aabb_size, const thrust::host_vector<int>& h_aabb_pid){
  std::ofstream outfile(filename, std::ios::binary);
  outfile.write((char*)&n_aabbs, sizeof(int));
  outfile.write((char*)&aabb_size, sizeof(int));
  outfile.write((char*)&h_aabbs[0], n_aabbs * sizeof(OptixAabb));
  outfile.write((char*)&h_aabb_pid[0], 1LL * n_aabbs * aabb_size * sizeof(int));
  for(int i=0; i<10; i++) printf("(%f, %f, %f) - (%f, %f, %f)\n", h_aabbs[i].minX, h_aabbs[i].minY, h_aabbs[i].minZ, h_aabbs[i].maxX, h_aabbs[i].maxY, h_aabbs[i].maxZ);
  outfile.close();
}

void RT_Entry::read_aabbs(const char* filename, int &n_aabbs, thrust::host_vector<OptixAabb>& h_aabbs, int &aabb_size, thrust::host_vector<int>& h_aabb_pid){
  std::ifstream infile(filename, std::ios::binary);
  infile.read((char*)&n_aabbs, sizeof(int));
  infile.read((char*)&aabb_size, sizeof(int));
  h_aabbs.resize(n_aabbs);
  h_aabb_pid.resize(n_aabbs * aabb_size);
  infile.read((char*)&h_aabbs[0], n_aabbs * sizeof(OptixAabb));
  infile.read((char*)&h_aabb_pid[0], 1LL * n_aabbs * aabb_size * sizeof(int));
  infile.close();
}