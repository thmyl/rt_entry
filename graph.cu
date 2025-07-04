#include "graph.h"
#include "auto_tune_bloom.h"
#include "warpselect/structure_on_device.cuh"
#include "graph_search.cuh"
#include <thrust/unique.h>

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
             std::string data_name_, std::string &data_path_, std::string &query_path_, std::string &gt_path_, std::string &graph_path_, int ALGO_, int search_width_, int topk_, int max_iter_){
  rt_entry = new RT_Entry(data_name_, n_subspaces_, buffer_size_, max_hits_, expand_ratio_, point_ratio_, n_candidates_);
  point_ratio = point_ratio_;
  n_hits = max_hits_;
  max_iter = max_iter_;
  datafile = (char*)data_path_.c_str();
  queryfile = (char*)query_path_.c_str();
  gtfile = (char*)gt_path_.c_str();
  graphfile = (char*)graph_path_.c_str();
  data_name = data_name_;
  rotation_matrix_path = data_name + "/O.fbin";
  mean_matrix_path = data_name + "/mean.fbin";
  pca_base_path = data_name + "/pca_base.fbin";
  
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

  if(ALGO == 0){
    d_points_.resize(h_points_.size());
    thrust::copy(h_points_.begin(), h_points_.end(), d_points_.begin());
  }
  
  d_queries_.resize(h_queries_.size());
  thrust::copy(h_queries_.begin(), h_queries_.end(), d_queries_.begin());

  d_gt_.resize(h_gt_.size());
  thrust::copy(h_gt_.begin(), h_gt_.end(), d_gt_.begin());

  rt_entry->set_size(DIM, np, nq, gt_k);
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
  CopyHostToDevice(h_pca_points, d_pca_points, np, dim_, DIM);
  // printf("reading PCA DIM file...\n");
  // thrust::host_vector<float> h_pca_points_DIM;
  // // std::string pca_base_DIM_path = data_name + "/pca_base_partly.fbin";
  // std::string pca_base_DIM_path = data_name + "/pca_base.fbin";
  // file_read::read_data(pca_base_DIM_path.c_str(), t_n, t_d, h_pca_points_DIM);
  // assert(t_n == np && t_d == DIM);
  // d_pca_points.resize(h_pca_points_DIM.size());
  // printf("copying pca points...\n");
  // thrust::copy(h_pca_points_DIM.begin(), h_pca_points_DIM.end(), d_pca_points.begin());
  // printf("finish copying pca points\n");
  // h_pca_points_DIM.resize(0);
  //debug end
  
  thrust::host_vector<float> h_rotation;
  file_read::read_data(rotation_matrix_path.c_str(), t_n, t_d, h_rotation);
  assert(t_n == dim_ && t_d == dim_);
  // d_rotation.resize(h_rotation.size());
  // thrust::copy(h_rotation.begin(), h_rotation.end(), d_rotation.begin());
  CopyHostToDevice(h_rotation, d_rotation, dim_, dim_, DIM);

  
  thrust::host_vector<float> h_mean;
  file_read::read_data(mean_matrix_path.c_str(), t_n, t_d, h_mean);
  assert(t_n == 1 && t_d == dim_);
  thrust::device_vector<float> d_mean;
  d_mean.resize(h_mean.size());
  thrust::copy(h_mean.begin(), h_mean.end(), d_mean.begin());

  
  //mr = mean * rotation
  thrust::device_vector<float> d_mr_row;
  d_mr_row.resize(1*dim_);
  float alpha = 1.0, beta = 0.0;
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "!!!! CUBLAS initialization error\n";
    return;
  }
  matrixMultiply(handle, d_mean, d_rotation, d_mr_row, t_n, DIM, dim_, alpha, beta);
  

  cublasDestroy(handle);
  // Timing::startTiming("replicateVector");
  d_pca_queries.resize(nq * DIM);


  auto* vec_ptr = thrust::raw_pointer_cast(d_mr_row.data());
  auto* result_ptr = thrust::raw_pointer_cast(d_pca_queries.data());

  replicateVector(result_ptr, vec_ptr, nq, DIM);


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
}

void Graph::Search(){
  Timing::startTiming("search");

  //----- pca projection -----
  if(ALGO == 1 || ALGO == 2){
    #ifdef DETAIL
      Timing::startTiming("pca projection");
    #endif
    float alpha = 1.0, beta = -1.0;
    matrixMultiply(handle_, d_queries_, d_rotation, d_pca_queries, nq, DIM, dim_, alpha, beta);
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
    rt_entry->Search(d_pca_points, d_pca_queries, d_gt_, d_entries, d_entries_dist, n_entries);
    Timing::stopTiming();
  }
  //----- TODO: graph search -----
    Timing::startTiming("graph search");
    GraphSearch();
    Timing::stopTiming();

  Timing::stopTiming(2);
  // if(ALGO == 1) check_entries(d_gt_);
  #ifdef REORDER
    thrust::copy(h_results.begin(), h_results.end(), d_results.begin());
  #endif
  cublasDestroy(handle_);
  check_results(d_gt_);
}

void Graph::GraphSearch(){
  int hash_len, bit, hash;
  hash_parameter(n_candidates, hash_len, bit, hash);
  constexpr int WARP_SIZE = 32;

  //变量名
  float *d_points_ptr;
  float *d_queries_ptr;
  if(ALGO == 1 || ALGO == 2){
    d_points_ptr = thrust::raw_pointer_cast(d_pca_points.data());
    d_queries_ptr = thrust::raw_pointer_cast(d_pca_queries.data());
  }
  else{
    d_points_ptr = thrust::raw_pointer_cast(d_points_.data());
    d_queries_ptr = thrust::raw_pointer_cast(d_queries_.data());
  }
  auto *d_results_ptr = thrust::raw_pointer_cast(d_results.data());
  auto *d_graph_ptr = thrust::raw_pointer_cast(d_graph_.data());

  auto *d_hits = thrust::raw_pointer_cast((rt_entry->subspaces_[0]).hits.data());
  auto *d_entries_ptr = thrust::raw_pointer_cast(rt_entry->subspaces_[0].aabb_entries.data());
  int aabb_size = rt_entry->subspaces_[0].aabb_size;

  auto *d_candidates_ptr = thrust::raw_pointer_cast(d_candidates.data());

  GraphSearchKernel<int, float, WARP_SIZE><<<nq, 64, ((search_width << offset_shift_) + n_candidates) * sizeof(KernelPair<float, int>)>>>
    (d_points_ptr, d_queries_ptr, d_results_ptr, d_graph_ptr, d_candidates_ptr, np,
    offset_shift_, n_candidates, topk, search_width, d_entries_ptr,
    d_hits, max_iter, ALGO);
  cudaDeviceSynchronize();

  #ifdef REORDER
  Timing::startTiming("reorder");
    thrust::copy(d_candidates.begin(), d_candidates.end(), h_candidates.begin());
    auto *h_candidates_ptr = thrust::raw_pointer_cast(h_candidates.data());
    auto *h_results_ptr = thrust::raw_pointer_cast(h_results.data());
    auto *h_points_ptr = thrust::raw_pointer_cast(h_points_.data());
    auto *h_queries_ptr = thrust::raw_pointer_cast(h_queries_.data());
    auto *candidates_dist_ptr = thrust::raw_pointer_cast(candidates_dist.data());
    parallel_reorder(h_candidates_ptr, h_results_ptr, n_candidates, topk, dim_, nq, h_points_ptr, np, h_queries_ptr, candidates_dist_ptr);
  Timing::stopTiming(2);
  #endif
}

void Graph::CleanUp(){
  if(ALGO == 1) rt_entry->CleanUp();
  // cublasDestroy(handle_);
}