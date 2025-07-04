#include "head.h"
#include "entry.h"
#include "warpselect/WarpSelect.cuh"

/*__global__ void calcDistance(int *d_candidates, float *d_candidates_dist, int *d_n_candidates, int buffer_size, int nq, float* d_query, float* d_data, int *hits, int *n_hits_per_query, int *hits_offset, int max_hits, int *aabb_pid, int *prefix_sum, int n_aabbs){
  int t_id = threadIdx.x;
  int b_id = blockIdx.x;
  int warp_size = 32;
  int n_warp = blockDim.x / warp_size;
  if(b_id < nq * max_hits){
    int q_id = b_id / max_hits;
    int hit_id = b_id % max_hits;
    if(hit_id >= n_hits_per_query[q_id]) return;
    int aabb_id = hits[q_id * max_hits + hit_id];

    int lane_id = t_id % warp_size;
    int warp_id = t_id / warp_size;

    int start = prefix_sum[aabb_id];
    int end = prefix_sum[aabb_id + 1];
    int num = end - start;
    for(int i=warp_id; i<num; i+=n_warp){
      int p_id = aabb_pid[start + i];
      //用32个lane计算距离

    // read d_query
      #if ENTRY_DIM > 0
        float q1 = 0;
        if (lane_id < ENTRY_DIM) {
          q1 = d_query[q_id * DIM + lane_id];
        }
      #endif
      #if ENTRY_DIM > 32
        float q2 = 0;
        if (lane_id + 32 < ENTRY_DIM) {
          q2 = d_query[q_id * DIM + lane_id + 32];
        }
      #endif
      #if ENTRY_DIM > 64
        float q3 = 0;
        if (lane_id + 64 < ENTRY_DIM) {
          q3 = d_query[q_id * DIM + lane_id + 64];
        }
      #endif
      #if ENTRY_DIM > 96
        float q4 = 0;
        if (lane_id + 96 < ENTRY_DIM) {
          q4 = d_query[q_id * DIM + lane_id + 96];
        }
      #endif

    // read d_data
      #if ENTRY_DIM > 0
        float p1 = 0;
        if (lane_id < ENTRY_DIM) {
          p1 = d_data[p_id * DIM + lane_id];
        }
      #endif
      #if ENTRY_DIM > 32
        float p2 = 0;
        if (lane_id + 32 < ENTRY_DIM) {
          p2 = d_data[p_id * DIM + lane_id + 32];
        }
      #endif
      #if ENTRY_DIM > 64
        float p3 = 0;
        if (lane_id + 64 < ENTRY_DIM) {
          p3 = d_data[p_id * DIM + lane_id + 64];
        }
      #endif
      #if ENTRY_DIM > 96
        float p4 = 0;
        if (lane_id + 96 < ENTRY_DIM) {
          p4 = d_data[p_id * DIM + lane_id + 96];
        }
      #endif

    // calculate distance
      #if ENTRY_DIM > 0
        float delta1 = (p1 - q1) * (p1 - q1);
      #endif
      #if ENTRY_DIM > 32
        float delta2 = (p2 - q2) * (p2 - q2);
      #endif
      #if ENTRY_DIM > 64
        float delta3 = (p3 - q3) * (p3 - q3);
      #endif
      #if ENTRY_DIM > 96
        float delta4 = (p4 - q4) * (p4 - q4);
      #endif
    
    // reduce
      float dist = 0;
      #if ENTRY_DIM > 0
        dist += delta1;
      #endif
      #if ENTRY_DIM > 32
        dist += delta2;
      #endif
      #if ENTRY_DIM > 64
        dist += delta3;
      #endif
      #if ENTRY_DIM > 96
        dist += delta4;
      #endif

      dist += __shfl_down_sync(FULL_MASK, dist, 16);
      dist += __shfl_down_sync(FULL_MASK, dist, 8);
      dist += __shfl_down_sync(FULL_MASK, dist, 4);
      dist += __shfl_down_sync(FULL_MASK, dist, 2);
      dist += __shfl_down_sync(FULL_MASK, dist, 1);

    // write
      if(lane_id == 0){
        int offset = i + hits_offset[q_id * max_hits + hit_id];
        if(offset < buffer_size){
          d_candidates_dist[q_id * buffer_size + offset] = dist;
          d_candidates[q_id * buffer_size + offset] = p_id;
        }
      }
    }
  }
}

__global__ void selectTopk(int k_, int *d_entries, float *d_entries_dist, int *d_candidates, float *d_candidates_dist, int *d_n_candidates, int buffer_size, int nq, int *hits, int *n_hits_per_query, int max_hits, int *aabb_pid, int *prefix_sum, int n_aabbs){
  int t_id = threadIdx.x;
  int b_id = blockIdx.x;
  int warp_size = 32;
  int n_warp = blockDim.x / warp_size;
  if(b_id < nq){
    int q_id = b_id;
    int num = 0;
    for(int i=0; i<n_hits_per_query[q_id]; i++){
      int aabb_id = hits[q_id * max_hits + i];
      int start = prefix_sum[aabb_id];
      int end = prefix_sum[aabb_id + 1];
      num += end - start;
    }
    num = min(buffer_size, num);
    d_n_candidates[q_id] = num;

    int lane_id = t_id % warp_size;
    int warp_id = t_id / warp_size;

    //求topk
    constexpr int WARP_SIZE = 32;
	  constexpr int NumWarpQ = 128;
	  constexpr int NumThreadQ = 4;
    WarpSelect<float, float, false, Comparator<float>, NumWarpQ, NumThreadQ, WARP_SIZE> heap(MAX, nq, NumWarpQ);
    int* crt_candidates = d_candidates + q_id * buffer_size;
    float* crt_candidates_dist = d_candidates_dist + q_id * buffer_size;
    
    for(int i =0; i < (num + 31) / 32; i++){
      int unroll_id = i * 32 + lane_id;
      if(unroll_id < num){
        heap.addThreadQ(crt_candidates_dist[unroll_id], crt_candidates[unroll_id]);
      }
      heap.reduce();
    }
    heap.reduce();
    
    for(int i=lane_id; i<k_; i += warp_size){
      d_entries[q_id * k_ + i] = heap.warpV[i/warp_size];
      d_entries_dist[q_id * k_ + i] = heap.warpK[i/warp_size];
    }
    heap.reset();
  }
}

__global__ void calc_hits_offset(int nq, int max_hits, int* hits, int* n_hits_per_query, int* aabb_pid, int* prefix_sum, int* hits_offset){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < nq){
    hits_offset[tid*max_hits] = 0;
    for(int i=0; i<n_hits_per_query[tid]; i++){
      int aabb_id = hits[tid*max_hits + i];
      int num = prefix_sum[aabb_id + 1] - prefix_sum[aabb_id];
      hits_offset[tid*max_hits + i + 1] = hits_offset[tid*max_hits + i] + num;
    }
  }
}


void RT_Entry::collect_candidates_onesubspace(
                                         thrust::device_vector<float> &d_pca_points,
                                         thrust::device_vector<float> &d_pca_queries,
                                         thrust::device_vector<int> &d_entries,
                                         thrust::device_vector<float> &d_entries_dist,
                                         int n_entries,
                                         thrust::device_vector<int> &hits,
                                         thrust::device_vector<int> &n_hits_per_query,
                                         thrust::device_vector<int> &hits_offset,
                                         int &max_hits, 
                                         thrust::device_vector<int> &aabb_pid,
                                         thrust::device_vector<int> &prefix_sum,
                                         int &n_aabbs){
  printf("collecting candidates...\n");
  // thrust::fill(d_n_candidates.begin(), d_n_candidates.end(), 0);
  auto *d_candidates_ptr = thrust::raw_pointer_cast(d_candidates.data());
  auto *d_candidates_dist_ptr = thrust::raw_pointer_cast(d_candidates_dist.data());
  auto *d_n_candidates_ptr = thrust::raw_pointer_cast(d_n_candidates.data());
  auto *hits_ptr = thrust::raw_pointer_cast(hits.data());
  auto *n_hits_per_query_ptr = thrust::raw_pointer_cast(n_hits_per_query.data());
  auto *hits_offset_ptr = thrust::raw_pointer_cast(hits_offset.data());
  auto *aabb_pid_ptr = thrust::raw_pointer_cast(aabb_pid.data());
  auto *prefix_sum_ptr = thrust::raw_pointer_cast(prefix_sum.data());
  auto *d_queries_ptr = thrust::raw_pointer_cast(d_pca_queries.data());
  auto *d_points_ptr = thrust::raw_pointer_cast(d_pca_points.data());
  // auto *d_queries_ptr = thrust::raw_pointer_cast(d_queries_.data());
  // auto *d_points_ptr = thrust::raw_pointer_cast(d_points_.data());
  auto *d_entries_ptr = thrust::raw_pointer_cast(d_entries.data());
  auto *d_entries_dist_ptr = thrust::raw_pointer_cast(d_entries_dist.data());


  #ifdef DETAIL
    Timing::startTiming("hits offset");
  #endif
  calc_hits_offset<<<(nq+31)/32, 32>>>(nq, max_hits, hits_ptr, n_hits_per_query_ptr, aabb_pid_ptr, prefix_sum_ptr, hits_offset_ptr);
  CUDA_SYNC_CHECK();
  #ifdef DETAIL
    Timing::stopTiming();
  #endif
  
  #ifdef DETAIL
    Timing::startTiming("calcDistance");
  #endif
  calcDistance<<<nq * max_hits, 128>>>(d_candidates_ptr, d_candidates_dist_ptr, d_n_candidates_ptr, buffer_size, nq, d_queries_ptr, d_points_ptr, hits_ptr, n_hits_per_query_ptr, hits_offset_ptr, max_hits, aabb_pid_ptr, prefix_sum_ptr, n_aabbs);
  CUDA_SYNC_CHECK();
  #ifdef DETAIL
    Timing::stopTiming();
  #endif

  #ifdef DETAIL
    Timing::startTiming("selectTopk");
  #endif
  selectTopk<<<nq, 32>>>(n_entries, d_entries_ptr, d_entries_dist_ptr, d_candidates_ptr, d_candidates_dist_ptr, d_n_candidates_ptr, buffer_size, nq, hits_ptr, n_hits_per_query_ptr, max_hits, aabb_pid_ptr, prefix_sum_ptr, n_aabbs);
  CUDA_SYNC_CHECK();
  #ifdef DETAIL
    Timing::stopTiming();
  #endif

}*/

__global__ void check_candidates_kernel(int *d_candidates, int *d_n_candidates, int nq, int buffer_size, int *d_gt, int gt_k, float *d_recall_1, float *d_recall_10, float *d_recall_100){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < nq){
    int offset = tid * buffer_size;
    int n_candidates = d_n_candidates[tid];
    for(int i=0; i<gt_k; i++){
      int gt = d_gt[tid * gt_k + i];
      for(int j=0; j<n_candidates; j++){
        if(gt == d_candidates[offset + j]){
          if(i<1)d_recall_1[tid] += 1;
          if(i<10){
            d_recall_10[tid] += 1;
          }
          if(i<100)d_recall_100[tid] += 1;
          break;
        }
      }
    }
  }
}

void RT_Entry::check_candidates(thrust::device_vector<int> &d_gt_){
  printf("checking candidates...\n");
  thrust::host_vector<int> h_n_candidates(nq, 0);
  thrust::copy(d_n_candidates.begin(), d_n_candidates.end(), h_n_candidates.begin());
  float sum_candidates = 0;
  for(int i=0; i<nq; i++){
    sum_candidates += h_n_candidates[i];
  }
  sum_candidates = sum_candidates / nq;
  printf("average candidates = %f\n", sum_candidates);

  auto *d_candidates_ptr = thrust::raw_pointer_cast(d_candidates.data());
  auto *d_n_candidates_ptr = thrust::raw_pointer_cast(d_n_candidates.data());
  auto *d_gt_ptr = thrust::raw_pointer_cast(d_gt_.data());
  thrust::device_vector<float> d_recall_1(nq, 0);
  thrust::device_vector<float> d_recall_10(nq, 0);
  thrust::device_vector<float> d_recall_100(nq, 0);
  check_candidates_kernel<<<(nq + 255)/256, 256>>>(d_candidates_ptr, d_n_candidates_ptr, nq, buffer_size, d_gt_ptr, gt_k,
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
  outfile << "candidate recall:\n";
  outfile <<  "recall@1 = " << sum_1 << " ms\n";
  outfile <<  "recall@10 = " << sum_10 << " ms\n";
  outfile <<  "recall@100 = " << sum_100 << " ms\n" << std::flush;
  outfile.close();
}

__global__ void subspace_copy_kernel(float *d_dst, float *d_src, int offset, int n, int d){
  int b_id = blockIdx.x;
  int t_id = threadIdx.x;
  if(b_id < n){
    if(t_id < 3){
      d_dst[b_id*3 + t_id] = d_src[b_id * d + offset + t_id];
    }
  }
}

void RT_Entry::subspace_copy(thrust::device_vector<float3> &d_dst, thrust::device_vector<float> &d_src, int offset){
  auto *d_dst_ptr = thrust::raw_pointer_cast(d_dst.data());
  auto *d_src_ptr = thrust::raw_pointer_cast(d_src.data());
  float* d_dst_ptr_ = reinterpret_cast<float*>(d_dst_ptr);
  subspace_copy_kernel<<<nq, 32>>>(d_dst_ptr_, d_src_ptr, offset, nq, dim_);
  CUDA_SYNC_CHECK();
}