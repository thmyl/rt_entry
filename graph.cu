#include "graph.h"
#include "auto_tune_bloom.h"
#include "warpselect/structure_on_device.cuh"
#include "warpselect/WarpSelect.cuh"
#include "graph_search.cuh"
#include "cache/page_cache.h"
#include <thrust/unique.h>
#include <numeric>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <limits>

extern "C" __global__ void topk_warp_kernel(
  const float* __restrict__ dists,
  int nq,
  int n_cluster,
  int K,
  int* __restrict__ out_topk
);

__global__ void mapIdKernel(int *d_unique, int unique_size, int *d_map_id){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < unique_size){
    d_map_id[d_unique[tid]] = tid;
  }
}

// 常量内存中的 cluster_to_page，仅在 size <= MAX_CLUSTER_TO_PAGE 时启用
// __constant__ int d_cluster_to_page[MAX_CLUSTER_TO_PAGE];
// int h_cluster_to_page_size = 0;
// __constant__ int d_cluster_to_page_size;

#ifdef USE_CACHE
#ifdef ENABLE_CONSTANT_CLUSTER_MAP
// 整表初始化常量映射
extern "C" void init_cluster_const_table(const int* h_data, int size){
  h_cluster_to_page_size = size;
  if (size <= 0 || size > MAX_CLUSTER_TO_PAGE) {
    int zero = 0;
    cudaMemcpyToSymbol(d_cluster_to_page_size, &zero, sizeof(int));
    return;
  }
  cudaMemcpyToSymbol(d_cluster_to_page_size, &size, sizeof(int));
  cudaMemcpyToSymbol(d_cluster_to_page, h_data, size * sizeof(int));
}

// 单元素增量更新常量映射（由 PageCache 调用）
extern "C" void update_cluster_const_entry(int idx, const int* h_value_ptr) {
  if (idx < 0) return;
  if (h_cluster_to_page_size <= 0 || idx >= h_cluster_to_page_size) return;
  cudaMemcpyToSymbol(d_cluster_to_page,
                     h_value_ptr,
                     sizeof(int),
                     static_cast<size_t>(idx) * sizeof(int),
                     cudaMemcpyHostToDevice);
}
#endif
#endif

Graph::~Graph(){
  // rt_entry->CleanUp();
}
extern void check_gpu_memory();

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
  d_query_batch_ids = nullptr;
  page_cache = nullptr;
}

void Graph::Init_entry(){
  rt_entry->BlockUp();
  rt_entry->InitRT();
  // d_entries.resize(nq * n_entries);
  // d_entries_dist.resize(nq * n_entries);
}

void Graph::Input(){
  std::cout<<"Input"<<std::endl;
  #ifdef DETAIL
    std::cout<<"Reading data_file: "<<datafile<<" ..."<<std::endl;
  #endif
  file_read::read_data(datafile, np, dim_, h_points_);
  #ifdef DETAIL
    std::cout<<"Reading query_file: "<<queryfile<<" ..."<<std::endl;
  #endif
  file_read::read_data(queryfile, nq, dim_, h_queries_);
  #ifdef DETAIL
    std::cout<<"Reading gt_file: "<<gtfile<<" ..."<<std::endl;
  #endif
  file_read::read_ivecs_file(gtfile, nq, gt_k, h_gt_);

  #ifdef USE_CACHE
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
      d_centroids_matrix.resize(h_centroids_matrix.size());
      thrust::copy(h_centroids_matrix.begin(), h_centroids_matrix.end(), d_centroids_matrix.begin());

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
      d_centroid_norms.resize(n_cluster);
      thrust::copy(centroid_norms_host.begin(), centroid_norms_host.end(), d_centroid_norms.begin());
      // printf("centroid_norms_host size = %d\n", centroid_norms_host.size());
      std::cout<<"centroid_norms_host size = "<<centroid_norms_host.size()<<std::endl;
    }
  
    dim_partial = PARTIAL_DIM;
    std::cout<<"dim_partial = "<<dim_partial<<std::endl;
    page_cache = new PageCache(page_size, n_page, dim_partial, n_cluster, np);
    load_linear_params();
  #endif

  std::cout<<"Input: copy points to device"<<std::endl;
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
  std::cout<<"Input: finish copy points to device"<<std::endl;
  
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
      std::cout<<"reading graph..."<<std::endl;
    #endif
    file_read::read_graph(graphfile, np, degree, h_graph_);
    // #ifdef DETAIL
    //   printf("graph size = %d\n", h_graph_.size());
    // #endif
    d_graph_.resize(h_graph_.size());
    thrust::copy(h_graph_.begin(), h_graph_.end(), d_graph_.begin());

    offset_shift_ = ceil(log(degree) / log(2));
    #ifdef DETAIL
      std::cout<<"offset_shift_ = "<<offset_shift_<<std::endl;
    #endif
  }
  std::cout<<"RB_Graph: finish read or build graph"<<std::endl;
}

void Graph::Projection(){
  std::cout<<"Projection: reading pca base file "<<pca_base_path<<std::endl;
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
  std::cout<<"read PCA file"<<std::endl;
  int t_n, t_d;
  thrust::host_vector<float> h_pca_points;
  file_read::read_data(pca_base_path.c_str(), t_n, t_d, h_pca_points);
  assert(t_n == np && t_d == dim_);
  // d_pca_points.resize(h_pca_points.size());
  // thrust::copy(h_pca_points.begin(), h_pca_points.end(), d_pca_points.begin());
  
  //debug begin
  //只拷贝前DIM维
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
  
  #ifdef USE_CACHE
    page_cache->init_data(h_pca_points.data(), dim_, DIM,
                          cluster_data.labels, cluster_data.cluster_points);
    page_cache->random_fill_cache();
  #endif

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
      std::cout<<"setting pca points..."<<std::endl;
    #endif
    rt_entry->set_pca_points(h_pca_points, dim_);
    #ifdef DETAIL
      std::cout<<"finish setting pca points"<<std::endl;
    #endif
  }
  
  #ifdef DETAIL
    std::cout<<"finish projection"<<std::endl;
  #endif
  preheat_cublas(nq, DIM, dim_);
  //将d_centroids_matrix转置
  std::cout<<"Projection: finish reading pca base file"<<std::endl;
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

template <int ThreadsPerBlock, int NumWarpQ, int NumThreadQ>
__global__ void query_cluster_top_warpselect_kernel(const float* __restrict__ dists,
                                                    int nq,
                                                    int n_cluster,
                                                    int K,
                                                    int* __restrict__ out_topk) {
    static_assert(ThreadsPerBlock % 32 == 0, "ThreadsPerBlock 必须是 32 的倍数");
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = ThreadsPerBlock / kWarpSize;

    int warp_id_in_block = threadIdx.x / kWarpSize;
    int lane_id = threadIdx.x & (kWarpSize - 1);
    int global_warp_id = blockIdx.x * kWarpsPerBlock + warp_id_in_block;
    if (global_warp_id >= nq) {
        return;
    }

    using Selector = WarpSelect<float,
                                int,
                                false,
                                Comparator<float>,
                                NumWarpQ,
                                NumThreadQ,
                                ThreadsPerBlock>;

    Selector selector(std::numeric_limits<float>::infinity(), -1, K);
    const float* row = dists + static_cast<size_t>(global_warp_id) * n_cluster;
    int iterations = (n_cluster + kWarpSize - 1) / kWarpSize;

    for (int it = 0; it < iterations; ++it) {
        int cid = it * kWarpSize + lane_id;
        float dist = (cid < n_cluster) ? row[cid] : std::numeric_limits<float>::infinity();
        int id = (cid < n_cluster) ? cid : -1;
        selector.add(dist, id);
    }

    selector.reduce();

    int out_base = global_warp_id * K;
    for (int i = 0; i < Selector::kNumWarpQRegisters; ++i) {
        int idx = i * kWarpSize + lane_id;
        if (idx < K) {
            out_topk[out_base + idx] = selector.warpV[i];
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

  matrixMultiplyABT(handle_, d_queries_, d_centroids_matrix, d_query_centroid_dists,
                 nq, n_cluster, dim_, alpha, beta);

  computeRowNorms(thrust::raw_pointer_cast(d_queries_.data()),
                  thrust::raw_pointer_cast(d_query_norms.data()),
                  nq, dim_);

  addNormsToDistances(thrust::raw_pointer_cast(d_query_centroid_dists.data()),
                      thrust::raw_pointer_cast(d_query_norms.data()),
                      thrust::raw_pointer_cast(d_centroid_norms.data()),
                      nq, n_cluster);

  d_query_top_clusters.resize(static_cast<size_t>(nq) * cluster_top_t);

  constexpr int kBlockSize = 256;
  constexpr int kWarpQueueSize = 256;
  constexpr int kThreadQueueSize = 4;

  if (cluster_top_t > kWarpQueueSize) {
    throw std::runtime_error("cluster_top_t 超过 WarpSelect 内部容量限制");
  }

  int warpsPerBlock = kBlockSize / 32;
  int gridSize = (nq + warpsPerBlock - 1) / warpsPerBlock;

  query_cluster_top_warpselect_kernel<kBlockSize, kWarpQueueSize, kThreadQueueSize><<<gridSize, kBlockSize>>>(
      thrust::raw_pointer_cast(d_query_centroid_dists.data()),
      nq,
      n_cluster,
      cluster_top_t,
      thrust::raw_pointer_cast(d_query_top_clusters.data()));

  CUDA_CHECK(cudaGetLastError());

  h_query_top_clusters.resize(static_cast<size_t>(nq) * cluster_top_t);
  thrust::copy(d_query_top_clusters.begin(), d_query_top_clusters.end(), 
               h_query_top_clusters.begin());

  thrust::device_vector<float>().swap(d_query_centroid_dists);
  thrust::device_vector<float>().swap(d_query_norms);
}

#ifdef USE_CACHE
  #ifndef GROUP_QUERY_BATCHES
  void Graph::build_query_batches() {
    batch_cluster_ids.clear();
    query_batch_ids.clear();
    if (cluster_top_t <= 0 || nq == 0) {
      return;
    }

    int total_batches = (nq + batch_size - 1) / batch_size;
    batch_cluster_ids.resize(total_batches);
    // query_batch_ids.resize(total_batches);
    query_batch_ids.resize(nq);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query_batch_ids), nq * sizeof(int)));

    for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
      int start = batch_idx * batch_size;
      int count = std::min(batch_size, nq - start);
      std::unordered_set<int> cluster_set;
      for (int i = 0; i < count; ++i) {
        int q = start + i;
        // query_batch_ids[batch_idx].push_back(q);
        query_batch_ids[q] = q;
        // if(q<10)std::cout<<"q = "<<q<<std::endl;
        for (int k = 0; k < cluster_top_t; ++k) {
          int cid = h_query_top_clusters[static_cast<size_t>(q) * cluster_top_t + k];
          // if(q<10)std::cout<<"cid = "<<cid<<std::endl;
          cluster_set.insert(cid);
        }
      }
      batch_cluster_ids[batch_idx] = std::vector<int>(cluster_set.begin(), cluster_set.end());
      // CUDA_CHECK(cudaMemcpy(d_query_batch_ids + start, query_batch_ids[batch_idx].data(), count * sizeof(int), cudaMemcpyHostToDevice));
      // printf("batch_idx = %d, cluster_set.size() = %d\n", batch_idx, cluster_set.size());
    }
    CUDA_CHECK(cudaMemcpy(d_query_batch_ids, query_batch_ids.data(), nq * sizeof(int), cudaMemcpyHostToDevice));
  }
  #endif
  #ifdef GROUP_QUERY_BATCHES
  struct QueryInfo {
    int qid;
    int rep;  // top-1 cluster
  };

  std::vector<QueryInfo> queries_to_sort;
  void Graph::build_query_batches() {
    batch_cluster_ids.clear();
    query_batch_ids.clear();
    if (cluster_top_t <= 0 || nq == 0) {
        return;
    }

    // 计算 batch 总数
    int total_batches = (nq + batch_size - 1) / batch_size;
    batch_cluster_ids.resize(total_batches);
    query_batch_ids.resize(nq);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query_batch_ids), nq * sizeof(int)));
    // -----------------------------
    // Step 1: 为每个 query 创建排序键（代表 cluster）
    // -----------------------------
    // struct QueryInfo {
    //     int qid;
    //     int rep;  // top-1 cluster
    // };

    // std::vector<QueryInfo> queries_to_sort(nq);
    queries_to_sort.resize(nq);
    for (int q = 0; q < nq; ++q) {
        int top1 = h_query_top_clusters[(size_t)q * cluster_top_t + 0];
        queries_to_sort[q] = {q, top1};
    }

    // -----------------------------
    // Step 2: 按 top-1 cluster 排序
    // -----------------------------
    std::sort(queries_to_sort.begin(), queries_to_sort.end(),
        [](const QueryInfo &a, const QueryInfo &b) {
            return a.rep < b.rep;
        }
    );

    // -----------------------------
    // Step 3: 顺序划分 batch
    // -----------------------------
    int qpos = 0;
    for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {

        int count = std::min(batch_size, nq - qpos);
        std::unordered_set<int> cluster_set;

        // 遍历该 batch 中的 query
        for (int i = 0; i < count; ++i) {
            int qid = queries_to_sort[qpos + i].qid;
            query_batch_ids[qpos + i] = qid;

            // 收集 top-t cluster
            for (int k = 0; k < cluster_top_t; ++k) {
                int cid = h_query_top_clusters[(size_t)qid * cluster_top_t + k];
                cluster_set.insert(cid);
            }
        }

        // 保存 batch 的 cluster 并集
        batch_cluster_ids[batch_idx] =
            std::vector<int>(cluster_set.begin(), cluster_set.end());

        qpos += count;
    }
    CUDA_CHECK(cudaMemcpy(d_query_batch_ids, query_batch_ids.data(), nq * sizeof(int), cudaMemcpyHostToDevice));
  }
  #endif
#else
  void Graph::build_query_batches() {
    query_batch_ids.clear();
    if (cluster_top_t <= 0 || nq == 0) {
      return;
    }

    int total_batches = (nq + batch_size - 1) / batch_size;
    // query_batch_ids.resize(total_batches);
    query_batch_ids.resize(nq);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query_batch_ids), nq * sizeof(int)));

    for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
      int start = batch_idx * batch_size;
      int count = std::min(batch_size, nq - start);
      for (int i = 0; i < count; ++i) {
        int q = start + i;
        // query_batch_ids[batch_idx].push_back(q);
        query_batch_ids[q] = q;
      }
    }
    CUDA_CHECK(cudaMemcpy(d_query_batch_ids, query_batch_ids.data(), nq * sizeof(int), cudaMemcpyHostToDevice));
  }
#endif

void Graph::build_query_batches_gpu() {
  if (cluster_top_t <= 0 || nq == 0) return;

  int total_batches = (nq + batch_size - 1) / batch_size;
  batch_cluster_ids.clear();
  batch_cluster_ids.resize(total_batches);

  // 1. 拷贝 top cluster 到 GPU
  thrust::device_vector<int> d_query_top(h_query_top_clusters.begin(), h_query_top_clusters.begin() + nq * cluster_top_t);

  for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
      int start = batch_idx * batch_size;
      int count = std::min(batch_size, nq - start);

      // 2. 将 batch 内所有 query 的 top cluster 拼成一个连续数组
      thrust::device_vector<int> d_batch_clusters(count * cluster_top_t);
      thrust::copy(
          d_query_top.begin() + start * cluster_top_t,
          d_query_top.begin() + (start + count) * cluster_top_t,
          d_batch_clusters.begin()
      );

      // 3. GPU 上 sort + unique 去重
      thrust::sort(d_batch_clusters.begin(), d_batch_clusters.end());
      auto new_end = thrust::unique(d_batch_clusters.begin(), d_batch_clusters.end());

      // 4. 拷贝回 CPU
      std::vector<int> h_batch(d_batch_clusters.begin(), new_end);
      batch_cluster_ids[batch_idx] = std::move(h_batch);
  }
}

void Graph::prefetch_batch_clusters(int batch_index, int query_offset, int batch_count, 
                                    cudaStream_t stream) {
  if(cluster_top_t <= 0 || !page_cache){
    return;
  }
  int copied_pages_prev = page_cache->copied_pages;
  int global_stream_count = 0;
  // int copy_count = 0;
  int copy_count = -100000000;//不停止拷贝
  for(int k=cluster_top_t-1; k>=0; --k){
    for(int q_i = query_offset; q_i < query_offset + batch_count; ++q_i){
      int q_id = query_batch_ids[q_i];
      int c_id = h_query_top_clusters[(size_t)q_id * cluster_top_t + k];
      // page_cache->prefetch_cluster(c_id, prefetch_streams->at(global_stream_count % prefetch_streams->size()));
      page_cache->prefetch_cluster(c_id, copy_count, stream);
      global_stream_count++;
    }
  }
  // for(int k=0; k<cluster_top_t; ++k){
  //   for(int q_i = query_offset; q_i < query_offset + batch_count; ++q_i){
  //     int q_id = query_batch_ids[q_i];
  //     int c_id = h_query_top_clusters[(size_t)q_id * cluster_top_t + k];
  //     page_cache->prefetch_cluster(c_id, copy_count, stream);
  //     if(copy_count >= page_cache->get_num_pages()) {
  //       return;
  //     }
  //     global_stream_count++;
  //   }
  // }
  if(copied_pages_prev != page_cache->copied_pages) {
    page_cache->update_map(stream);
  }
}

void Graph::Search(){
  // Timing::startTiming("search");
  // printf("batch_size = %d\n", batch_size);
  std::cout<<"batch_size = "<<batch_size<<std::endl;

  // ======================= create stream =======================
  cudaStream_t graph_stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking));

  #ifdef USE_CACHE
    // 创建 prefetch stream pool 用于并行预取多个 cluster
    std::vector<cudaStream_t> prefetch_streams;
    if (cluster_top_t > 0) {
      const int num_prefetch_streams = 2;  // 限制 stream 数量
      prefetch_streams.resize(num_prefetch_streams, nullptr);
      for (int i = 0; i < num_prefetch_streams; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&prefetch_streams[i], cudaStreamNonBlocking));
      }
    }
  #endif

  Timing::startTiming("search");
  // ======================= build query batches =======================
  #ifdef USE_CACHE
    if (cluster_top_t > 0) {
      Timing::startTiming("compute_query_cluster_top");
      compute_query_cluster_top();
      Timing::stopTiming(2);

      Timing::startTiming("build_query_batches");
      build_query_batches();
      Timing::stopTiming(2);
    }
  #else
    Timing::startTiming("build_query_batches");
    build_query_batches();
    Timing::stopTiming(2);
  #endif
  

  // ======================= pca projection =======================
  if(ALGO == 1 || ALGO == 2){
    #ifdef DETAIL
      Timing::startTiming("pca projection");
    #endif
    // d_pca_queries.resize(static_cast<size_t>(nq) * DIM);
    float alpha = 1.0, beta = -1.0;
    std::cout<<"matrixMultiply"<<std::endl;
    matrixMultiply(handle_, d_queries_, d_rotation, d_pca_queries_full, nq, dim_, dim_, alpha, beta);
    std::cout<<"finish matrixMultiply"<<std::endl;
    
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
  // ======================= rt search =======================
    Timing::startTiming("search_entry");
    rt_entry->Search(d_pca_points, d_pca_queries_full, d_gt_, d_entries, d_entries_dist, n_entries);
    Timing::stopTiming(2);
  }
  // ======================= graph search =======================
    Timing::startTiming("graph search");

    // #region define variables
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

      // float* d_query_batch = d_queries_ptr + static_cast<size_t>(query_offset) * query_dim;
      auto *d_results_ptr = thrust::raw_pointer_cast(d_results.data());
      // int* d_results_batch = d_results_ptr + static_cast<size_t>(query_offset) * topk;
      auto *d_graph_ptr = thrust::raw_pointer_cast(d_graph_.data());

      auto *d_hits_all = thrust::raw_pointer_cast((rt_entry->subspaces_[0]).hits.data());
      // int* d_hits_batch = d_hits_all ? d_hits_all + query_offset : nullptr;
      auto *d_entries_ptr = thrust::raw_pointer_cast(rt_entry->subspaces_[0].aabb_entries.data());

      auto *d_candidates_ptr = thrust::raw_pointer_cast(d_candidates.data());
      // int* d_candidates_batch = d_candidates_ptr + static_cast<size_t>(query_offset) * n_candidates;

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
    // #endregion define variables

    int flag = 0; //交替拷贝
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    
    int total_batches = (nq + batch_size - 1) / batch_size;
    // std::cout<<"total_batches = "<<total_batches<<std::endl;
    for(int batch_idx = 0; batch_idx < total_batches; ++batch_idx){
      int start = batch_idx * batch_size;
      int count = std::min(batch_size, nq - start);
      #ifdef USE_CACHE
        if(cluster_top_t > 0 && total_batches > 0){
          prefetch_batch_clusters(batch_idx, start, count, prefetch_streams[flag]);
          // cudaDeviceSynchronize();
          //stream同步
          CUDA_CHECK(cudaEventRecord(event, prefetch_streams[flag]));
          CUDA_CHECK(cudaStreamWaitEvent(graph_stream, event, 0));
        }
      #endif
      // upload_cluster_to_page(page_cache->cluster_to_page.data(), page_cache->total_cluster_pages);
      // GraphSearchBatch(start, count, graph_stream);
      GraphSearchKernel<int, float, WARP_SIZE><<<count, 64, shared_mem, graph_stream>>>
      (d_points_ptr, d_queries_ptr, /*d_results_batch*/ d_results_ptr, d_graph_ptr, /*d_candidates_batch*/ d_candidates_ptr, np,
      start, d_query_batch_ids,
      offset_shift_, n_candidates, topk, search_width, d_entries_ptr,
      /*d_hits_batch*/ d_hits_all, max_iter, ALGO,
      d_point_infos, d_cluster_to_page, cache_ptr, page_size,
      dim_partial, dim_,
      d_query_top_ptr, cluster_top_t,
      d_linear_w_ptr, d_linear_b_ptr, linear_params_dim);
      // cudaDeviceSynchronize();
      CUDA_CHECK(cudaStreamSynchronize(graph_stream));
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

  #ifdef DETAIL
    #ifdef USE_CACHE
      if (page_cache) {
        std::cout<<"copied_pages = "<<page_cache->copied_pages<<std::endl;
      }
    #endif
  #endif

  #ifdef USE_CACHE
    // 同步并销毁 prefetch streams
    if (!prefetch_streams.empty()) {
      for (auto& stream : prefetch_streams) {
        if (stream != nullptr) {
          CUDA_CHECK(cudaStreamSynchronize(stream));
          CUDA_CHECK(cudaStreamDestroy(stream));
        }
      }
    }
  #endif
  CUDA_CHECK(cudaEventDestroy(event));
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

  // float* d_query_batch = d_queries_ptr + static_cast<size_t>(query_offset) * query_dim;
  auto *d_results_ptr = thrust::raw_pointer_cast(d_results.data());
  // int* d_results_batch = d_results_ptr + static_cast<size_t>(query_offset) * topk;
  auto *d_graph_ptr = thrust::raw_pointer_cast(d_graph_.data());

  auto *d_hits_all = thrust::raw_pointer_cast((rt_entry->subspaces_[0]).hits.data());
  // int* d_hits_batch = d_hits_all ? d_hits_all + query_offset : nullptr;
  auto *d_entries_ptr = thrust::raw_pointer_cast(rt_entry->subspaces_[0].aabb_entries.data());

  auto *d_candidates_ptr = thrust::raw_pointer_cast(d_candidates.data());
  // int* d_candidates_batch = d_candidates_ptr + static_cast<size_t>(query_offset) * n_candidates;

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
    (d_points_ptr, d_queries_ptr, /*d_results_batch*/ d_results_ptr, d_graph_ptr, /*d_candidates_batch*/ d_candidates_ptr, np,
    query_offset, d_query_batch_ids,
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
  if(d_query_batch_ids) {
    cudaFree(d_query_batch_ids);
    d_query_batch_ids = nullptr;
  }
  // cublasDestroy(handle_);
}