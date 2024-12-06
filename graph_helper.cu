#include "graph.h"
__global__ void check_entries_kernel(uint *d_entries, uint n_entries, uint nq, uint *d_gt, uint gt_k, float *d_recall_1, float *d_recall_10, float *d_recall_100){
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < nq){
    uint offset = tid * n_entries;
    for(int i=0; i<gt_k; i++){
      uint gt = d_gt[tid * gt_k + i];
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

void Graph::check_entries(thrust::device_vector<uint> &d_gt_){
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


__global__ void check_results_kernel(uint *d_results, uint n_results, uint nq, uint *d_gt, uint gt_k, float *d_recall_1, float *d_recall_10, float *d_recall_100){
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < nq){
    uint offset = tid * n_results;
    for(int i=0; i<gt_k; i++){
      uint gt = d_gt[tid * gt_k + i];
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

void Graph::check_results(thrust::device_vector<uint> &d_gt_){
  printf("checking results...\n");

  auto *d_results_ptr = thrust::raw_pointer_cast(d_results.data());
  auto *d_gt_ptr = thrust::raw_pointer_cast(d_gt_.data());
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
  outfile << "results recall:\n";
  outfile <<  "recall@1 = " << sum_1 << " ms\n";
  outfile <<  "recall@10 = " << sum_10 << " ms\n";
  outfile <<  "recall@100 = " << sum_100 << " ms\n" << std::flush;
  outfile.close();
}