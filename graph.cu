#include "graph.h"
#include "auto_tune_bloom.h"
#include "warpselect/structure_on_device.cuh"
#include "graph_search.cuh"
#include "cache/page_cache.h"
#include <thrust/unique.h>
#include <numeric>
#include <unordered_set>
#include <algorithm>
#include <fstream>

__global__ void mapIdKernel(int *d_unique, int unique_size, int *d_map_id){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < unique_size){
    d_map_id[d_unique[tid]] = tid;
  }
}

Graph::~Graph(){
  // rt_entry->CleanUp();
}
extern void check_gpu_memory();
// void check_gpu_memory() {
//   size_t free_memory, total_memory;
//   cudaMemGetInfo(&free_memory, &total_memory);
  
//   std::cout << "Free memory: " << free_memory / 1024 / 1024 << " MB" << std::endl;
//   std::cout << "Total memory: " << total_memory / 1024 / 1024 << " MB" << std::endl;
//   std::cout << "Used memory: " << (total_memory - free_memory) / 1024 / 1024 << " MB" << std::endl;

//   std::ofstream outfile;
//   outfile.open(OUTFILE, std::ios_base::app);
//   outfile <<  "Used memory: " << (total_memory - free_memory) / 1024 / 1024 << " MB\n" << std::flush;
//   outfile.close();
// }

Graph::Graph(int n_subspaces_, int buffer_size_, int n_candidates_, int max_hits_, double expand_ratio_, double point_ratio_,
             std::string data_name_, std::string &data_path_, std::string &query_path_, std::string &gt_path_, std::string &centroids_path_, std::string &graph_path_, int ALGO_, int search_width_, int topk_, int max_iter_, 
             int t_, int n_cluster_, int page_size_, int n_page_){
  rt_entry = new RT_Entry(data_name_, n_subspaces_, buffer_size_, max_hits_, expand_ratio_, point_ratio_, n_candidates_);
  point_ratio = point_ratio_;
  n_hits = max_hits_;
  max_iter = max_iter_;
  datafile = (char*)data_path_.c_str();
  queryfile = (char*)query_path_.c_str();
  gtfile = (char*)gt_path_.c_str();
  graphfile = (char*)graph_path_.c_str();
  centroids_file = (char*)centroids_path_.c_str();
  
  data_name = data_name_;
  rotation_matrix_path = data_name + "/rotation.fbin";
  mean_matrix_path = data_name + "/mean.fbin";
  pca_base_path = data_name + "/rotated_base.fbin";
  linear_params_path = data_name + "/linear_params.bin";
  
  page_size = page_size_;
  n_page = n_page_;
  n_cluster = n_cluster_;
  cluster_top_t = t_;
  // n_entries = n_candidates_;
  n_candidates = n_candidates_;
  topk = topk_;
  
  ALGO = ALGO_;
  search_width = search_width_;
  n_candidates = pow(2.0, ceil(log(n_candidates)/log(2)));
  if(topk == 0) topk = n_candidates;
  #ifdef DETAIL
    printf("n_candidates = %d\n", n_candidates);
  #endif

  cublasStatus_t status = cublasCreate(&handle_);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "!!!! CUBLAS initialization error\n";
    return;
  }
}

void Graph::Init_entry(){
  rt_entry->BlockUp();
  rt_entry->InitRT();
  // d_entries.resize(nq * n_entries);
  // d_entries_dist.resize(nq * n_entries);
}

void Graph::Input(){
  #ifdef DETAIL
    printf("Reading data_file: %s ...\n", datafile);
  #endif
  file_read::read_data(datafile, np, dim_, h_points_);
  #ifdef DETAIL
    printf("Reading query_file: %s ...\n", queryfile);
  #endif
  file_read::read_data(queryfile, nq, dim_, h_queries_);
  #ifdef DETAIL
    printf("Reading gt_file: %s ...\n", gtfile);
  #endif
  file_read::read_ivecs_file(gtfile, nq, gt_k, h_gt_);
  file_read::read_centroids(centroids_file, cluster_data, np, dim_);
  std::cout<<"cluster_top_t = "<<cluster_top_t<<std::endl;

  if (cluster_top_t > 0) {
    if (cluster_data.K != n_cluster) {
      n_cluster = cluster_data.K;
    }
    cluster_top_t = std::min(cluster_top_t, n_cluster);

    thrust::host_vector<float> h_centroids_matrix(
        cluster_data.centroids.begin(),
        cluster_data.centroids.end());
    std::cout<<"w1"<<std::endl;
    d_centroids_matrix.resize(h_centroids_matrix.size());
    thrust::copy(h_centroids_matrix.begin(), h_centroids_matrix.end(), d_centroids_matrix.begin());
    std::cout<<"w2"<<std::endl;

    std::vector<float> centroid_norms_host(n_cluster, 0.0f);
    for (int cid = 0; cid < n_cluster; ++cid) {
      const float* centroid = cluster_data.centroids.data() + static_cast<size_t>(cid) * dim_;
      double sum = 0.0;
      for (int j = 0; j < dim_; ++j) {
        double v = centroid[j];
        sum += v * v;
      }
      centroid_norms_host[cid] = static_cast<float>(sum);
    }
    std::cout<<"w3"<<std::endl;
    d_centroid_norms.resize(n_cluster);
    thrust::copy(centroid_norms_host.begin(), centroid_norms_host.end(), d_centroid_norms.begin());
    // printf("centroid_norms_host size = %d\n", centroid_norms_host.size());
    std::cout<<"centroid_norms_host size = "<<centroid_norms_host.size()<<std::endl;
  }
 
  dim_partial = dim_ - DIM;
  std::cout<<"dim_partial = "<<dim_partial<<std::endl;
  page_cache = new PageCache(page_size, n_page, dim_partial, n_cluster, np);
  std::cout<<"w4"<<std::endl;
  load_linear_params();

  if(ALGO == 0){
    d_points_.resize(h_points_.size());
    thrust::copy(h_points_.begin(), h_points_.end(), d_points_.begin());
  }
  
  d_queries_.resize(h_queries_.size());
  thrust::copy(h_queries_.begin(), h_queries_.end(), d_queries_.begin());

  d_gt_.resize(h_gt_.size());
  thrust::copy(h_gt_.begin(), h_gt_.end(), d_gt_.begin());

  rt_entry->set_size(dim_, np, nq, gt_k);
  d_results.resize(nq * topk);
  h_results.resize(nq * topk);
  n_entries = n_candidates;
  // d_entries.resize(nq * n_entries);
  // n_entries = point_ratio * np * n_hits;
  // n_entries = n_candidates;
  // if(ALGO==1) n_entries = point_ratio * np;
  
  #ifdef REORDER
    d_candidates.resize(nq * n_candidates);
    h_candidates.resize(nq * n_candidates);
    candidates_dist.resize(nq * n_candidates);
  #endif
}

void Graph::load_linear_params() {
  std::ifstream infile(linear_params_path, std::ios::binary);
  if (!infile.is_open()) {
    std::cerr << "Failed to open linear params file: " << linear_params_path << std::endl;
    return;
  }

  int file_dim = 0;
  infile.read(reinterpret_cast<char*>(&file_dim), sizeof(int));
  if (!infile) {
    std::cerr << "Failed to read linear params header" << std::endl;
    return;
  }

  linear_params_dim = std::min(file_dim, dim_);
  linear_w_host.resize(linear_params_dim+1);
  linear_b_host.resize(linear_params_dim+1);

  for (int i = 1; i <= file_dim; ++i) {
    float w = 0.0f, b = 0.0f;
    infile.read(reinterpret_cast<char*>(&w), sizeof(float));
    infile.read(reinterpret_cast<char*>(&b), sizeof(float));
    if (i <= linear_params_dim) {
      linear_w_host[i] = w;
      linear_b_host[i] = b;
    }
  }
  infile.close();

  d_linear_w.resize(linear_params_dim+1);
  d_linear_b.resize(linear_params_dim+1);
  thrust::copy(linear_w_host.begin(), linear_w_host.end(), d_linear_w.begin());
  thrust::copy(linear_b_host.begin(), linear_b_host.end(), d_linear_b.begin());
}

void Graph::RB_Graph(){
  //read or build graph
  FILE *graph_file = fopen(graphfile, "rb");
  if(graph_file == NULL){
    printf("building graph...\n");
    degree = FLAGS_degree;
    // TODO: BuildGraph();
  }
  else{
    fclose(graph_file);
    #ifdef DETAIL
      printf("reading graph...\n");
    #endif
    file_read::read_graph(graphfile, np, degree, h_graph_);
    // #ifdef DETAIL
    //   printf("graph size = %d\n", h_graph_.size());
    // #endif
    d_graph_.resize(h_graph_.size());
    thrust::copy(h_graph_.begin(), h_graph_.end(), d_graph_.begin());

    offset_shift_ = ceil(log(degree) / log(2));
    #ifdef DETAIL
      printf("offset_shift_ = %d\n", offset_shift_);
    #endif
  }
}

void Graph::Projection(){
  FILE *pca_base_file = fopen(pca_base_path.c_str(), "rb");
  FILE *rotation_matrix_file = fopen(rotation_matrix_path.c_str(), "rb");
  if(pca_base_file == NULL){
    PCA pca(h_points_.data(), np, dim_);
    if(rotation_matrix_file == NULL){
      printf("computing PCA matrix...\n");
      pca.calc_eigenvalues();//计算mean和rotation
      pca.save_mean_rotation(mean_matrix_path.c_str(), rotation_matrix_path.c_str());
      int n_subspaces = rt_entry->get_n_subspaces();
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

  // h_points_.resize(0);

  // 读取文件并计算points的投影
  printf("read PCA file\n");
  int t_n, t_d;
  thrust::host_vector<float> h_pca_points;
  file_read::read_data(pca_base_path.c_str(), t_n, t_d, h_pca_points);
  assert(t_n == np && t_d == dim_);
  // d_pca_points.resize(h_pca_points.size());
  // thrust::copy(h_pca_points.begin(), h_pca_points.end(), d_pca_points.begin());
  
  //debug begin
  //只拷贝前DIM维
  std::cout<<"w5"<<std::endl;
  // CopyHostToDevice(h_pca_points, d_pca_points, np, dim_, DIM);
  d_pca_points.resize(1LL*np*DIM);
  cudaMemcpy2D(
    thrust::raw_pointer_cast(d_pca_points.data()),   // 目标起始地址（nq × DIM）
    DIM * sizeof(float),                              // 目标每行跨度（字节）
    thrust::raw_pointer_cast(h_pca_points.data()),  // 源起始地址（np × dim_）
    dim_ * sizeof(float),                             // 源每行跨度（字节）
    DIM * sizeof(float),                              // 每行拷贝宽度（字节）
    np,                                               // 行数
    cudaMemcpyHostToDevice
  );

  // printf("reading PCA DIM file...\n");
  // thrust::host_vector<float> h_pca_points_DIM;
  // std::string pca_base_DIM_path = data_name + "/pca_base_partly.fbin";
  // // std::string pca_base_DIM_path = data_name + "/pca_base.fbin";
  // file_read::read_data(pca_base_DIM_path.c_str(), t_n, t_d, h_pca_points_DIM);
  // assert(t_n == np && t_d == DIM);
  // d_pca_points.resize(h_pca_points_DIM.size());
  // printf("copying pca points...\n");
  // thrust::copy(h_pca_points_DIM.begin(), h_pca_points_DIM.end(), d_pca_points.begin());
  // printf("finish copying pca points\n");
  // h_pca_points_DIM.resize(0);
  //debug end
  
  std::cout<<"w6"<<std::endl;
  page_cache->init_data(h_pca_points.data(), dim_, DIM,
                        cluster_data.labels, cluster_data.cluster_points);
  std::cout<<"w7"<<std::endl;

  thrust::host_vector<float> h_rotation;
  file_read::read_data(rotation_matrix_path.c_str(), t_n, t_d, h_rotation);
  assert(t_n == dim_ && t_d == dim_);
  d_rotation.resize(h_rotation.size());
  thrust::copy(h_rotation.begin(), h_rotation.end(), d_rotation.begin());
  // CopyHostToDevice(h_rotation, d_rotation, dim_, dim_, DIM);

  
  thrust::host_vector<float> h_mean;
  file_read::read_data(mean_matrix_path.c_str(), t_n, t_d, h_mean);
  assert(t_n == 1 && t_d == dim_);
  thrust::device_vector<float> d_mean;
  d_mean.resize(h_mean.size());
  thrust::copy(h_mean.begin(), h_mean.end(), d_mean.begin());

  
  //mr = mean * rotation
  thrust::device_vector<float> d_mr_row;
  d_mr_row.resize(dim_);
  float alpha = 1.0, beta = 0.0;
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "!!!! CUBLAS initialization error\n";
    return;
  }
  matrixMultiply(handle, d_mean, d_rotation, d_mr_row, t_n, dim_, dim_, alpha, beta);
  

  cublasDestroy(handle);
  // Timing::startTiming("replicateVector");
  d_pca_queries_full.resize(static_cast<size_t>(nq) * dim_);


  auto* vec_ptr = thrust::raw_pointer_cast(d_mr_row.data());
  auto* result_ptr_full = thrust::raw_pointer_cast(d_pca_queries_full.data());

  replicateVector(result_ptr_full, vec_ptr, nq, dim_);


  d_mr_row.resize(0);
  thrust::device_vector<float>().swap(d_mr_row);
  d_mean.resize(0);
  thrust::device_vector<float>().swap(d_mean);
  // Timing::stopTiming();


  if(ALGO==1){
    #ifdef DETAIL
      printf("setting pca points...\n");
    #endif
    rt_entry->set_pca_points(h_pca_points, dim_);
    #ifdef DETAIL
      printf("finish setting pca points\n");
    #endif
  }
  
  #ifdef DETAIL
    printf("finish projection\n");
  #endif
  preheat_cublas(nq, DIM, dim_);
  //将d_centroids_matrix转置
}

//
// 主 Kernel：每个 query 一个 block
//
__global__
void topk_dynamic_kernel(const float* __restrict__ dists,
                         int nq,
                         int n_cluster,
                         int K,                // top_t
                         int* __restrict__ out_topk)
{
    int q = blockIdx.x;
    if (q >= nq) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    extern __shared__ float shm[];   
    // 动态分配
    float* shm_vals = shm;                    // blockDim.x * K
    int*   shm_ids  = (int*)(shm + blockDim.x * K);

    // 每线程局部 top-K
    float* local_vals = shm_vals + tid * K;
    int*   local_ids  = shm_ids  + tid * K;

    // 初始化局部 K
    for (int i = 0; i < K; i++) {
        local_vals[i] = 1e30f;
        local_ids[i] = -1;
    }

    const float* row = dists + (size_t)q * n_cluster;
    int pos = 0;
    float maxv = local_vals[0];

    // 每个线程扫描自己的部分
    for (int idx = tid; idx < n_cluster; idx += stride) {
        float v = row[idx];
        // update_topk_dynamic(local_vals, local_ids, K, v, idx);
        if(v < maxv){
          local_vals[pos] = v;
          local_ids[pos] = idx;
          for(int j=0; j<K; j++){
            if(local_vals[j] > v){
              maxv = local_vals[j];
              pos = j;
            }
          }
        }
    }

    __syncthreads();

    // block 内合并 top-K（由一个线程完成）
    if (tid == 0) {
        for (int k = 0; k < K; k++) {
            float best_v = 1e30f;
            int best_id = -1;
            int best_pos = -1;

            int total = blockDim.x * K;
            for (int j = 0; j < total; j++) {
                if (shm_vals[j] < best_v) {
                    best_v = shm_vals[j];
                    best_id = shm_ids[j];
                    best_pos = j;
                }
            }

            // 记录第 k 个最小
            out_topk[q * K + k] = best_id;

            // 防止重复选择
            shm_vals[best_pos] = 1e30f;
        }
    }
}

void Graph::compute_query_cluster_top() {
  if (cluster_top_t <= 0 || n_cluster == 0) {
    return;
  }

  if (d_centroids_matrix.empty()) {
    return;
  }

  d_query_centroid_dists.resize(static_cast<size_t>(nq) * n_cluster);
  d_query_norms.resize(nq);

  float alpha = -2.0f;
  float beta = 0.0f;
  
  // Matrix multiplication: compute -2 * queries * centroids^T
  // No need to synchronize here - CUBLAS operations are queued
  matrixMultiplyABT(handle_, d_queries_, d_centroids_matrix, d_query_centroid_dists,
                 nq, n_cluster, dim_, alpha, beta);

  // Compute query norms
  computeRowNorms(thrust::raw_pointer_cast(d_queries_.data()),
                  thrust::raw_pointer_cast(d_query_norms.data()),
                  nq, dim_);

  // Add norms to distances: dist = -2*q*c^T + ||q||^2 + ||c||^2
  addNormsToDistances(thrust::raw_pointer_cast(d_query_centroid_dists.data()),
                      thrust::raw_pointer_cast(d_query_norms.data()),
                      thrust::raw_pointer_cast(d_centroid_norms.data()),
                      nq, n_cluster);

  // GPU-based top-k selection: avoid copying entire distance matrix to CPU
  // Use the existing topk_dynamic_kernel which is more efficient
  d_query_top_clusters.resize(static_cast<size_t>(nq) * cluster_top_t);
  
  int blockSize = 256;  // One block per query
  int gridSize = nq;
  // Dynamic shared memory: blockDim * K * (float + int)
  size_t shmBytes = static_cast<size_t>(blockSize) * cluster_top_t * (sizeof(float) + sizeof(int));
  
  topk_dynamic_kernel<<<gridSize, blockSize, shmBytes>>>(
    thrust::raw_pointer_cast(d_query_centroid_dists.data()),
    nq,
    n_cluster,
    cluster_top_t,
    thrust::raw_pointer_cast(d_query_top_clusters.data())
  );
  
  // Check for errors (only once at the end, no unnecessary synchronization)
  CUDA_CHECK(cudaGetLastError());
  
  // Copy to host only if needed (for build_query_batches)
  h_query_top_clusters.resize(static_cast<size_t>(nq) * cluster_top_t);
  thrust::copy(d_query_top_clusters.begin(), d_query_top_clusters.end(), 
               h_query_top_clusters.begin());

  // Clean up intermediate data
  thrust::device_vector<float>().swap(d_query_centroid_dists);
  thrust::device_vector<float>().swap(d_query_norms);
} 

void Graph::build_query_batches() {
  batch_cluster_ids.clear();
  if (cluster_top_t <= 0 || nq == 0) {
    return;
  }

  int total_batches = (nq + batch_size - 1) / batch_size;
  batch_cluster_ids.resize(total_batches);

  for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
    int start = batch_idx * batch_size;
    int count = std::min(batch_size, nq - start);
    std::unordered_set<int> cluster_set;
    for (int i = 0; i < count; ++i) {
      int q = start + i;
      // if(q<10)std::cout<<"q = "<<q<<std::endl;
      for (int k = 0; k < cluster_top_t; ++k) {
        int cid = h_query_top_clusters[static_cast<size_t>(q) * cluster_top_t + k];
        // if(q<10)std::cout<<"cid = "<<cid<<std::endl;
        cluster_set.insert(cid);
      }
    }
    batch_cluster_ids[batch_idx] = std::vector<int>(cluster_set.begin(), cluster_set.end());
  }
  // for(int i=0; i<10; i++){
  //   std::cout<<"batch_cluster_ids["<<i<<"] = ";
  //   for(int j=0; j<batch_cluster_ids[i].size(); j++){
  //     std::cout<<batch_cluster_ids[i][j]<<" ";
  //   }
  //   std::cout<<std::endl;
  // }
}

void Graph::prefetch_batch_clusters(int batch_index, int query_offset, int batch_count, 
                                    const std::vector<cudaStream_t>* prefetch_streams) {
  if (cluster_top_t <= 0 || !page_cache) {
    return;
  }

  (void)query_offset;
  (void)batch_count;

  if (batch_index < 0 || batch_index >= static_cast<int>(batch_cluster_ids.size())) {
    return;
  }

  const auto& clusters = batch_cluster_ids[batch_index];
  if (clusters.empty()) {
    return;
  }

  // 使用提供的 stream pool，round-robin 方式为每个 cluster 分配 stream
  // 这样多个 cluster 的 prefetch 操作可以在不同的 stream 上并行执行
  if (prefetch_streams && !prefetch_streams->empty()) {
    int num_streams = static_cast<int>(prefetch_streams->size());
    for (size_t i = 0; i < clusters.size(); ++i) {
      int cid = clusters[i];
      cudaStream_t stream = (*prefetch_streams)[i % num_streams];
      page_cache->prefetch_cluster(cid, stream);
    }
  } else {
    // 如果没有提供 stream pool，使用 page_cache 的默认 stream（向后兼容）
    cudaStream_t stream = page_cache->get_default_stream();
    for (int cid : clusters) {
      page_cache->prefetch_cluster(cid, stream);
    }
  }
}

void Graph::Search(){
  Timing::startTiming("search");
  // printf("batch_size = %d\n", batch_size);
  std::cout<<"batch_size = "<<batch_size<<std::endl;

  if (cluster_top_t > 0) {
    Timing::startTiming("compute_query_cluster_top");
    compute_query_cluster_top();
    Timing::stopTiming(2);
    Timing::startTiming("build_query_batches");
    build_query_batches();
    Timing::stopTiming(2);
  }

  //----- pca projection -----
  if(ALGO == 1 || ALGO == 2){
    #ifdef DETAIL
      Timing::startTiming("pca projection");
    #endif
    // d_pca_queries.resize(static_cast<size_t>(nq) * DIM);
    float alpha = 1.0, beta = -1.0;
    std::cout<<"matrixMultiply"<<std::endl;
    matrixMultiply(handle_, d_queries_, d_rotation, d_pca_queries_full, nq, dim_, dim_, alpha, beta);
    std::cout<<"finish matrixMultiply"<<std::endl;
    // thrust::copy_n(d_pca_queries_full.begin(), static_cast<size_t>(nq) * DIM, d_pca_queries.begin());
    // cudaMemcpy2D(
    //   thrust::raw_pointer_cast(d_pca_queries.data()),   // 目标起始地址（nq × DIM）
    //   DIM * sizeof(float),                              // 目标每行跨度（字节）
    //   thrust::raw_pointer_cast(d_pca_queries_full.data()), // 源起始地址（nq × dim_）
    //   dim_ * sizeof(float),                             // 源每行跨度（字节）
    //   DIM * sizeof(float),                              // 每行拷贝宽度（字节）
    //   nq,                                               // 行数
    //   cudaMemcpyDeviceToDevice
    // );
    #ifdef DETAIL
      Timing::stopTiming(2);
    #endif
    d_queries_.resize(0);
    // d_queries_.shrink_to_fit();
    thrust::device_vector<float>().swap(d_queries_);
    d_rotation.resize(0);
    // d_rotation.shrink_to_fit();
    thrust::device_vector<float>().swap(d_rotation);
  }

  if(ALGO == 1){
  //----- rt search -----
    Timing::startTiming("search_entry");
    rt_entry->Search(d_pca_points, d_pca_queries_full, d_gt_, d_entries, d_entries_dist, n_entries);
    Timing::stopTiming(2);
  }
  //----- TODO: graph search -----
    Timing::startTiming("graph search");
    cudaStream_t graph_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking));
    
    // 创建 prefetch stream pool 用于并行预取多个 cluster
    std::vector<cudaStream_t> prefetch_streams;
    if (cluster_top_t > 0) {
      const int num_prefetch_streams = 8;  // 限制 stream 数量
      prefetch_streams.resize(num_prefetch_streams, nullptr);
      for (int i = 0; i < num_prefetch_streams; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&prefetch_streams[i], cudaStreamNonBlocking));
      }
    }
    
    int total_batches = (nq + batch_size - 1) / batch_size;
    
    // 预取第一个 batch（如果启用）
    // if (cluster_top_t > 0 && total_batches > 0) {
    //   int first_start = 0;
    //   int first_count = std::min(batch_size, nq - first_start);
    //   prefetch_batch_clusters(0, first_start, first_count, &prefetch_streams);
    // }
    
    // for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
    //   // printf("batch_idx = %d\n", batch_idx);
    //   int start = batch_idx * batch_size;
    //   int count = std::min(batch_size, nq - start);
      
    //   // 执行当前 batch 的图搜索（与预取并行）
    //   GraphSearchBatch(start, count, graph_stream);
      
    //   // 同时预取下一个 batch（如果存在）
    //   if (cluster_top_t > 0 && batch_idx + 1 < total_batches) {
    //     int next_start = (batch_idx + 1) * batch_size;
    //     int next_count = std::min(batch_size, nq - next_start);
    //     prefetch_batch_clusters(batch_idx + 1, next_start, next_count, &prefetch_streams);
    //   }
    //   // cudaDeviceSynchronize();
    //   // CUDA_CHECK(cudaGetLastError());  
    //   // printf("finish batch_idx = %d\n", batch_idx);
    // }
    for(int batch_idx = 0; batch_idx < total_batches; ++batch_idx){
      int start = batch_idx * batch_size;
      int count = std::min(batch_size, nq - start);
      if(cluster_top_t > 0 && total_batches > 0){
        prefetch_batch_clusters(batch_idx, start, count, &prefetch_streams);
      }
      GraphSearchBatch(start, count, graph_stream);
    }
    
    // 同步所有 stream
    CUDA_CHECK(cudaStreamSynchronize(graph_stream));
    CUDA_CHECK(cudaStreamDestroy(graph_stream));
    
    // 同步并销毁 prefetch streams
    if (!prefetch_streams.empty()) {
      for (auto& stream : prefetch_streams) {
        if (stream != nullptr) {
          CUDA_CHECK(cudaStreamSynchronize(stream));
          CUDA_CHECK(cudaStreamDestroy(stream));
        }
      }
    }
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());  
    Timing::stopTiming(2);

  Timing::stopTiming(2);
  // if(ALGO == 1) check_entries(d_gt_);
  #ifdef REORDER
    thrust::copy(h_results.begin(), h_results.end(), d_results.begin());
  #endif
  cublasDestroy(handle_);
  check_results(d_gt_);
}

void Graph::GraphSearchBatch(int query_offset, int batch_count, cudaStream_t stream){
  if (batch_count <= 0) {
    return;
  }
  int hash_len, bit, hash;
  hash_parameter(n_candidates, hash_len, bit, hash);
  constexpr int WARP_SIZE = 32;

  float *d_points_ptr;
  float *d_queries_ptr;
  int query_dim;
  if(ALGO == 1 || ALGO == 2){
    d_points_ptr = thrust::raw_pointer_cast(d_pca_points.data());
    d_queries_ptr = thrust::raw_pointer_cast(d_pca_queries_full.data());
    query_dim = dim_;
  }
  else{
    d_points_ptr = thrust::raw_pointer_cast(d_points_.data());
    d_queries_ptr = thrust::raw_pointer_cast(d_queries_.data());
    query_dim = dim_;
  }

  float* d_query_batch = d_queries_ptr + static_cast<size_t>(query_offset) * query_dim;
  auto *d_results_ptr = thrust::raw_pointer_cast(d_results.data());
  int* d_results_batch = d_results_ptr + static_cast<size_t>(query_offset) * topk;
  auto *d_graph_ptr = thrust::raw_pointer_cast(d_graph_.data());

  auto *d_hits_all = thrust::raw_pointer_cast((rt_entry->subspaces_[0]).hits.data());
  int* d_hits_batch = d_hits_all ? d_hits_all + query_offset : nullptr;
  auto *d_entries_ptr = thrust::raw_pointer_cast(rt_entry->subspaces_[0].aabb_entries.data());

  auto *d_candidates_ptr = thrust::raw_pointer_cast(d_candidates.data());
  int* d_candidates_batch = d_candidates_ptr + static_cast<size_t>(query_offset) * n_candidates;

  const PointInfo* d_point_infos = page_cache ? page_cache->device_point_info_ptr() : nullptr;
  const int* d_cluster_to_page = page_cache ? page_cache->device_cluster_map() : nullptr;
  const float* cache_ptr = page_cache ? page_cache->device_cache_ptr() : nullptr;
  const float* d_query_full_ptr = d_pca_queries_full.empty() ? nullptr : thrust::raw_pointer_cast(d_pca_queries_full.data());
  const int* d_query_top_ptr = (!d_query_top_clusters.empty() && cluster_top_t > 0)
                                 ? thrust::raw_pointer_cast(d_query_top_clusters.data())
                                 : nullptr;
  const float* d_linear_w_ptr = linear_params_dim > 0 ? thrust::raw_pointer_cast(d_linear_w.data()) : nullptr;
  const float* d_linear_b_ptr = linear_params_dim > 0 ? thrust::raw_pointer_cast(d_linear_b.data()) : nullptr;

  size_t shared_mem = ((search_width << offset_shift_) + n_candidates) * sizeof(KernelPair<float, int>);
  GraphSearchKernel<int, float, WARP_SIZE><<<batch_count, 64, shared_mem, stream>>>
    (d_points_ptr, d_query_full_ptr, /*d_results_batch*/ d_results_ptr, d_graph_ptr, /*d_candidates_batch*/ d_candidates_ptr, np,
    query_offset,
    offset_shift_, n_candidates, topk, search_width, d_entries_ptr,
    /*d_hits_batch*/ d_hits_all, max_iter, ALGO,
    d_point_infos, d_cluster_to_page, cache_ptr, page_size,
    dim_partial, dim_,
    d_query_top_ptr, cluster_top_t,
    d_linear_w_ptr, d_linear_b_ptr, linear_params_dim);
  CUDA_CHECK(cudaGetLastError());  
}

void Graph::CleanUp(){
  if(ALGO == 1) rt_entry->CleanUp();
  // cublasDestroy(handle_);
}