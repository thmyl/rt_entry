#include "graph.h"
#include <immintrin.h>
#include <omp.h>

__global__ void check_entries_kernel(int *d_entries, int n_entries, int nq, int *d_gt, int gt_k, float *d_recall_1, float *d_recall_10, float *d_recall_100){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < nq){
    int offset = tid * n_entries;
    for(int i=0; i<gt_k; i++){
      int gt = d_gt[tid * gt_k + i];
      for(int j=0; j<n_entries; j++){
        if(gt == d_entries[offset + j]){
          if(i<1)d_recall_1[tid] += 1;
          if(i<10)d_recall_10[tid] += 1;
          if(i<100)d_recall_100[tid] += 1;
          break;
        }
      }
    }
  }
}

void Graph::check_entries(thrust::device_vector<int> &d_gt_){
  printf("checking entries...\n");

  auto *d_entries_ptr = thrust::raw_pointer_cast(d_entries.data());
  auto *d_gt_ptr = thrust::raw_pointer_cast(d_gt_.data());
  thrust::device_vector<float> d_recall_1(nq, 0);
  thrust::device_vector<float> d_recall_10(nq, 0);
  thrust::device_vector<float> d_recall_100(nq, 0);
  check_entries_kernel<<<(nq + 255)/256, 256>>>(d_entries_ptr, n_entries, nq, d_gt_ptr, gt_k,
                                                    thrust::raw_pointer_cast(d_recall_1.data()),
                                                    thrust::raw_pointer_cast(d_recall_10.data()),
                                                    thrust::raw_pointer_cast(d_recall_100.data()));
  CUDA_SYNC_CHECK();
  thrust::host_vector<float> h_recall_1 = d_recall_1;
  thrust::host_vector<float> h_recall_10 = d_recall_10;
  thrust::host_vector<float> h_recall_100 = d_recall_100;
  float sum_1 = 0;
  float sum_10 = 0;
  float sum_100 = 0;
  for(int i=0; i<nq; i++){
    sum_1 += h_recall_1[i];
    sum_10 += h_recall_10[i];
    sum_100 += h_recall_100[i];
  }
  sum_1 = sum_1 / nq;
  sum_10 = sum_10 / nq / 10;
  sum_100 = sum_100 / nq / 100;
  printf("recall@1 = %f\n", sum_1);
  printf("recall@10 = %f\n", sum_10);
  printf("recall@100 = %f\n", sum_100);

  std::ofstream outfile;
  outfile.open(OUTFILE, std::ios_base::app);
  outfile << "entries recall:\n";
  outfile <<  "recall@1 = " << sum_1 << " ms\n";
  outfile <<  "recall@10 = " << sum_10 << " ms\n";
  outfile <<  "recall@100 = " << sum_100 << " ms\n" << std::flush;
  outfile.close();
}


__global__ void check_results_kernel(int *d_results, int n_results, int nq, int *d_gt, int gt_k, float *d_recall_1, float *d_recall_10, float *d_recall_100){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < nq){
    int offset = tid * n_results;
    for(int i=0; i<gt_k; i++){
      int gt = d_gt[tid * gt_k + i];
      for(int j=0; j<n_results; j++){
        if(gt == d_results[offset + j]){
          if(i<1)d_recall_1[tid] += 1;
          if(i<10)d_recall_10[tid] += 1;
          if(i<100)d_recall_100[tid] += 1;
          break;
        }
      }
    }
  }
}

void Graph::check_results(thrust::device_vector<int> &d_gt_){
  printf("checking results...\n");

  auto *d_results_ptr = thrust::raw_pointer_cast(d_results.data());
  auto *d_gt_ptr = thrust::raw_pointer_cast(d_gt_.data());
  printf("gt_k = %d\n", gt_k);
  thrust::device_vector<float> d_recall_1(nq, 0);
  thrust::device_vector<float> d_recall_10(nq, 0);
  thrust::device_vector<float> d_recall_100(nq, 0);
  check_results_kernel<<<(nq + 255)/256, 256>>>(d_results_ptr, topk, nq, d_gt_ptr, gt_k,
                                                    thrust::raw_pointer_cast(d_recall_1.data()),
                                                    thrust::raw_pointer_cast(d_recall_10.data()),
                                                    thrust::raw_pointer_cast(d_recall_100.data()));
  CUDA_SYNC_CHECK();
  thrust::host_vector<float> h_recall_1 = d_recall_1;
  thrust::host_vector<float> h_recall_10 = d_recall_10;
  thrust::host_vector<float> h_recall_100 = d_recall_100;
  float sum_1 = 0;
  float sum_10 = 0;
  float sum_100 = 0;
  for(int i=0; i<nq; i++){
    sum_1 += h_recall_1[i];
    sum_10 += h_recall_10[i];
    sum_100 += h_recall_100[i];
  }
  sum_1 = sum_1 / nq;
  sum_10 = sum_10 / nq / 10;
  sum_100 = sum_100 / nq / 100;
  printf("recall@1 = %f\n", sum_1);
  printf("recall@10 = %f\n", sum_10);
  printf("recall@100 = %f\n", sum_100);

  std::ofstream outfile;
  outfile.open(OUTFILE, std::ios_base::app);
  // outfile << "results recall:\n";
  // outfile <<  "recall@1 = " << sum_1 << " ms\n";
  outfile <<  "recall@10 = " << sum_10 << " ms\n" << std::flush;
  // outfile <<  "recall@100 = " << sum_100 << " ms\n" << std::flush;
  outfile.close();
}

bool cmp(Pair a, Pair b){
  return a.dist < b.dist;
}

/*void Graph::paralled_reorder(int* candidates, int* results, int n_candidates, int topk, int dim_, int nq, float* points, int np, float* queries, Pair* candidates_dist){
  for(int q_id=0; q_id<nq; q_id++){
    Pair* cur_candidates_dist = candidates_dist + q_id * n_candidates;
    for(int j=0; j<n_candidates; j++){
      int p_id = candidates[q_id*n_candidates+j];
      cur_candidates_dist[j].id = p_id;
      float dis = 0;
      for(int d=0; d<dim_; d++){
        dis = dis + (points[p_id*dim_+d] - queries[q_id*dim_+d]) * (points[p_id*dim_+d] - queries[q_id*dim_+d]);
      }
      cur_candidates_dist[j].dist = dis;
    }
    std::sort(cur_candidates_dist, cur_candidates_dist+n_candidates, cmp);
    for(int j=0; j<topk; j++){
      results[q_id*topk+j] = cur_candidates_dist[j].id;
    }
  }
}*/

void Graph::parallel_reorder(int* candidates, int* results, int n_candidates, int topk, int dim_, int nq, float* points, int np, float* queries, Pair* candidates_dist) {
  // int max_threads = omp_get_max_threads();
  int max_threads = 64;

  #pragma omp parallel for schedule(dynamic) num_threads(max_threads)
  for (int q_id_ = 0; q_id_ < nq; ++q_id_) {
    int q_id = q_id_;
    Pair* cur_candidates_dist = candidates_dist + q_id * n_candidates;

    // 计算距离并填充cur_candidates_dist
    // for (int j = 0; j < n_candidates; ++j) {
    //   int p_id = candidates[q_id * n_candidates + j];
    //   cur_candidates_dist[j].id = p_id;
    //   float dis = 0.0f;
    //   for (int d = 0; d < dim_; ++d) {
    //     float diff = points[p_id * dim_ + d] - queries[q_id * dim_ + d];
    //     dis += diff * diff;
    //   }
    //   cur_candidates_dist[j].dist = dis;
    // }

    // 计算距离并填充cur_candidates_dist
    for (int j = 0; j < n_candidates; ++j) {
      long long p_id = candidates[q_id * n_candidates + j];
      // p_id = j;
      cur_candidates_dist[j].id = p_id;
      __m256 dis_vec = _mm256_setzero_ps(); // 初始化为0

      // 使用SIMD计算距离
      for (int d = 0; d < dim_; d += 8) {
        // 加载points和queries的数据到向量寄存器
        __m256 point_vec = _mm256_loadu_ps(&points[p_id * dim_ + d]);
        __m256 query_vec = _mm256_loadu_ps(&queries[q_id * dim_ + d]);

        // 计算差值
        __m256 diff_vec = _mm256_sub_ps(point_vec, query_vec);

        // 计算平方
        __m256 square_vec = _mm256_mul_ps(diff_vec, diff_vec);

        // 累加到dis_vec
        dis_vec = _mm256_add_ps(dis_vec, square_vec);
      }

      // 将向量结果汇总为标量
      float dis = 0.0f;
      float temp_dis[8];
      _mm256_storeu_ps(temp_dis, dis_vec); // 存储向量结果到数组
      for (int k = 0; k < 8 && (k + 8 * ((dim_ - 1) / 8)) < dim_; ++k) {
        dis += temp_dis[k];
      }

      // 处理剩余的维度（如果dim_不是8的倍数）
      for (int d = (dim_ / 8) * 8; d < dim_; ++d) {
        float diff = points[p_id * dim_ + d] - queries[q_id * dim_ + d];
        dis += diff * diff;
      }

      cur_candidates_dist[j].dist = dis;
    }

    std::sort(cur_candidates_dist, cur_candidates_dist + n_candidates, cmp);

    for (int j = 0; j < topk; ++j) {
      results[q_id * topk + j] = cur_candidates_dist[j].id;
    }
  }
}

void Graph::CopyHostToDevice(thrust::host_vector<float> &h_data, thrust::device_vector<float> &d_data, int n, int d, int d_){
  d_data.resize(1LL*n*d_);
  for(int i=0; i<n; i++){
    thrust::copy(h_data.begin() + 1LL*i*d, h_data.begin()+ 1LL*i*d + d_, d_data.begin() + 1LL*i*d_);
  }
}