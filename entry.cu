#include <iostream>
#include <cstdio>
#include "raytracing.h"
#include "entry.h"
#include "pca.h"
#include "aabb.h"
// #include "casting_kernels.h"

void Entry::Input(char* datafile, char* queryfile, char* gtfile){
  printf("Reading data_file: %s ...\n", datafile);
  file_read::read_data(datafile, np, dim_, h_points_);
  printf("Reading query_file: %s ...\n", queryfile);
  file_read::read_data(queryfile, nq, dim_, h_queries_);
  printf("Reading gt_file: %s ...\n", gtfile);
  file_read::read_ivecs_file(gtfile, nq, gt_k, h_gt_);

  d_points_.resize(h_points_.size());
  thrust::copy(h_points_.begin(), h_points_.end(), d_points_.begin());
  d_queries_.resize(h_queries_.size());
  thrust::copy(h_queries_.begin(), h_queries_.end(), d_queries_.begin());
  d_gt_.resize(h_gt_.size());
  thrust::copy(h_gt_.begin(), h_gt_.end(), d_gt_.begin());

  subspaces_.resize(n_subspaces);
  h_candidates.resize(nq * buffer_size);
  d_candidates.resize(nq * buffer_size);
  d_candidates_dist.resize(nq * buffer_size);
  h_n_candidates.resize(nq);
  d_n_candidates.resize(nq);
  d_entries.resize(nq * entries_size);
  d_entries_dist.resize(nq * entries_size);
}

void Entry::Projection(){
  FILE *pca_base_file = fopen(pca_base_path.c_str(), "rb");
  FILE *rotation_matrix_file = fopen(rotation_matrix_path.c_str(), "rb");
  if(pca_base_file == NULL){
    PCA pca(h_points_.data(), np, dim_);
    if(rotation_matrix_file == NULL){
      printf("computing PCA matrix...\n");
      pca.calc_eigenvalues();//计算mean和rotation
      pca.save_mean_rotation(mean_matrix_path.c_str(), rotation_matrix_path.c_str());
      printf("the first %d ratio = %f\n", n_subspaces*3, pca.Ratio(n_subspaces*3));
    }
    else {
      fclose(rotation_matrix_file);
      printf("reading PCA matrix...\n");
      pca.read_mean_rotation(mean_matrix_path.c_str(), rotation_matrix_path.c_str());
    }
    pca.calc_result(dim_);
    pca.save_result(dim_, pca_base_path.c_str());
  }
  else 
    fclose(pca_base_file);

  // 读取文件并计算points的投影
  printf("read PCA file\n");
  uint t_n, t_d;
  // thrust::host_vector<float> h_pca_points;
  file_read::read_data(pca_base_path.c_str(), t_n, t_d, h_pca_points);
  assert(t_n == np && t_d == dim_);
  d_pca_points.resize(h_pca_points.size());
  thrust::copy(h_pca_points.begin(), h_pca_points.end(), d_pca_points.begin());
  
  // thrust::host_vector<float> h_rotation;
  file_read::read_data(rotation_matrix_path.c_str(), t_n, t_d, h_rotation);
  assert(t_n == dim_ && t_d == dim_);
  d_rotation.resize(h_rotation.size());
  thrust::copy(h_rotation.begin(), h_rotation.end(), d_rotation.begin());
  
  // thrust::host_vector<float> h_mean;
  file_read::read_data(mean_matrix_path.c_str(), t_n, t_d, h_mean);
  assert(t_n == 1 && t_d == dim_);
  d_mean.resize(h_mean.size());
  thrust::copy(h_mean.begin(), h_mean.end(), d_mean.begin());
  
  //mr = mean * rotation
  d_mr_row.resize(1*dim_);
  float alpha = 1.0, beta = 0.0;
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "!!!! CUBLAS initialization error\n";
    return;
  }
  matrixMultiply(handle, d_mean, d_rotation, d_mr_row, 1, dim_, dim_, alpha, beta);
  cublasDestroy(handle);
  Timing::startTiming("replicateVector");
  d_pca_queries.resize(nq * dim_);
  replicateVector(d_pca_queries, d_mr_row, nq, dim_);
  Timing::stopTiming();

  for(int space = 0; space < n_subspaces; space++){
    subspaces_[space].h_points.resize(np);
    for(int i = 0; i < np; i++){
      uint offset = i*dim_ + space * 3;
      subspaces_[space].h_points[i] = float3{h_pca_points[offset], h_pca_points[offset + 1], h_pca_points[offset + 2]};
    }
    // subspaces_[space].h_rotation.resize(3*dim_);
    // for(int i = 0; i < dim_; i++){
    //   for(int j=0; j<3; j++){
    //     subspaces_[space].h_rotation[i*3 + j] = h_rotation[i*dim_ + space*3 + j];
    //   }
    // }
  }

  for(int space = 0; space < n_subspaces; space++){
    auto &d_points = subspaces_[space].d_points;
    // auto &d_rotation = subspaces_[space].d_rotation;
    auto &h_points = subspaces_[space].h_points;
    // auto &h_rotation = subspaces_[space].h_rotation;
    d_points.resize(h_points.size());
    thrust::copy(h_points.begin(), h_points.end(), d_points.begin());
    // d_rotation.resize(h_rotation.size());
    // thrust::copy(h_rotation.begin(), h_rotation.end(), d_rotation.begin());
  }
}

void Entry::BlockUp(){
  printf("BlockUp function\n");
  for(int space = 0; space < n_subspaces; space++){
    //均匀分配块
    auto &world_bounds = subspaces_[space].world_bounds;
    auto &h_points = subspaces_[space].h_points;
    auto &d_points = subspaces_[space].d_points;
    auto &h_aabbs = subspaces_[space].h_aabbs;
    auto &d_aabbs = subspaces_[space].d_aabbs;
    auto &n_aabbs = subspaces_[space].n_aabbs;
    auto &aabb_pid = subspaces_[space].aabb_pid;
    auto &prefix_sum = subspaces_[space].prefix_sum;

    computeMinMax(world_bounds, h_points);
    printf("world bound of space%d: (%f, %f, %f) - (%f, %f, %f)\n", space, 
           world_bounds.minX, world_bounds.minY, world_bounds.minZ, 
           world_bounds.maxX, world_bounds.maxY, world_bounds.maxZ);
#ifdef GRID
    // uint grid_size = 13;
    n_aabbs = grid_size * grid_size * grid_size;
    generate_uniform_aabbs(h_aabbs, grid_size, world_bounds, n_aabbs);
    d_aabbs.resize(n_aabbs);
    thrust::copy(h_aabbs.begin(), h_aabbs.end(), d_aabbs.begin());

    prefix_sum.resize(n_aabbs + 1);
    aabb_pid.resize(np);
    TraversePointsInRegion(d_points, d_aabbs, aabb_pid, prefix_sum);

    //expand aabb in unform
    float expand_ratio = expand_ratio;
    float3 delta = make_float3(world_bounds.maxX - world_bounds.minX, world_bounds.maxY - world_bounds.minY, world_bounds.maxZ - world_bounds.minZ);
    delta.x = delta.x / grid_size; delta.y = delta.y / grid_size; delta.z = delta.z / grid_size;
    float3 expand = make_float3(expand_ratio * delta.x, expand_ratio * delta.y, expand_ratio * delta.z);
    expandAabb_uniform(h_aabbs, expand);
    thrust::copy(h_aabbs.begin(), h_aabbs.end(), d_aabbs.begin());
#elif defined(KDTREE)
    float3* h_points_ptr = thrust::raw_pointer_cast(subspaces_[space].h_points.data());
    Kdtree kdtree(h_points_ptr, np, np * point_ratio, world_bounds, h_aabbs);
    n_aabbs = h_aabbs.size();
    printf("n_aabbs = %d\n", n_aabbs);
    d_aabbs.resize(n_aabbs);
    thrust::copy(h_aabbs.begin(), h_aabbs.end(), d_aabbs.begin());

    prefix_sum.resize(n_aabbs + 1);
    aabb_pid.resize(np);
    // TraversePointsInRegion(d_points, d_aabbs, aabb_pid, prefix_sum);
    kdtree.computeAabbPid(aabb_pid, prefix_sum, n_aabbs);

    expandAabb(h_aabbs, expand_ratio);
    thrust::copy(h_aabbs.begin(), h_aabbs.end(), d_aabbs.begin());
#endif
  }
}

void Entry::InitRT(){
  for(int space = 0; space < n_subspaces; space++){
    auto &rt = subspaces_[space].rt;
    auto &d_aabbs = subspaces_[space].d_aabbs;
    auto &n_aabbs = subspaces_[space].n_aabbs;
    auto *d_aabbs_ptr = thrust::raw_pointer_cast(d_aabbs.data());
    rt.Setup();
    rt.BuildAccel(d_aabbs_ptr, n_aabbs);

    subspaces_[space].hits.resize(nq * max_hits);
    subspaces_[space].n_hits_per_query.resize(nq);
  }
}

void Entry::Search(){
  Timing::startTiming("search");
  //pca投影
  /*Timing::startTiming("pca projection");
  thrust::device_vector<float> d_queries_mz = d_queries_;
  subtraction(thrust::raw_pointer_cast(d_queries_mz.data()), thrust::raw_pointer_cast(d_mean.data()), nq, dim_);
  for(int space = 0; space < n_subspaces; space++){
    auto &d_queries = subspaces_[space].d_queries;
    auto &d_rotation = subspaces_[space].d_rotation;
    d_queries.resize(nq);
    rotate(thrust::raw_pointer_cast(d_queries.data()), thrust::raw_pointer_cast(d_queries_mz.data()), thrust::raw_pointer_cast(d_rotation.data()), nq, dim_);
  }
  Timing::stopTiming();*/
  Timing::startTiming("pca projection");
  float alpha = 1.0, beta = -1.0;
  matrixMultiply(handle_, d_queries_, d_rotation, d_pca_queries, nq, dim_, dim_, alpha, beta);
  Timing::stopTiming(2);

  //rt搜索
  Timing::startTiming("rt search");
  for(uint space = 0; space < n_subspaces; space++){
    auto &rt = subspaces_[space].rt;
    auto &hits = subspaces_[space].hits;
    auto &n_hits_per_query = subspaces_[space].n_hits_per_query;
    rt.search(thrust::raw_pointer_cast(d_pca_queries.data()), nq, space*3, dim_, thrust::raw_pointer_cast(hits.data()), thrust::raw_pointer_cast(n_hits_per_query.data()), max_hits);
    
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
  Timing::stopTiming(2);

  //合并搜索结果
  Timing::startTiming("collect candidates");
  if(n_subspaces == 1){
    // Timing::startTiming("before collect candidates onesubspace");
    auto &hits = subspaces_[0].hits;
    auto &n_hits_per_query = subspaces_[0].n_hits_per_query;
    auto &aabb_pid = subspaces_[0].aabb_pid;
    auto &prefix_sum = subspaces_[0].prefix_sum;
    auto &n_aabbs = subspaces_[0].n_aabbs;
    // Timing::stopTiming();

    // Timing::startTiming("collect candidates onesubspace");
    collect_candidates_onesubspace(hits, n_hits_per_query, max_hits, aabb_pid, prefix_sum, n_aabbs);
    // Timing::stopTiming();
  }
  Timing::stopTiming(2);
  Timing::stopTiming(2);
  check_candidates();
  check_entries();
}

void Entry::CleanUp(){
  for(int space = 0; space < n_subspaces; space++){
    auto &rt = subspaces_[space].rt;
    rt.CleanUp();
  }
  cublasDestroy(handle_);
}


