#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace std;

void read_fvecs(ifstream& file, float*& dataset, int& n, int& d){
  file.read((char*)&d, 4);
  file.seekg(0, ios::end);
  long long size = file.tellg();
  n = size / (4 + 4 * d);
  file.seekg(0, ios::beg);
  printf("n = %d, d = %d\n", n, d);
  dataset = new float[1LL*n * d];
  for(int i = 0; i < n; i++){
    file.seekg(4, ios::cur);
    file.read((char*)(dataset + 1LL*i * d), 4 * d);
  }
}

void read_ivecs(ifstream& file, int*& dataset, int& n, int& d){
  file.read((char*)&d, 4);
  file.seekg(0, ios::end);
  long long size = file.tellg();
  n = size / (4 + 4 * d);
  file.seekg(0, ios::beg);
  printf("n = %d, d = %d\n", n, d);
  dataset = new int[1LL*n * d];
  for(int i = 0; i < n; i++){
    file.seekg(4, ios::cur);
    file.read((char*)(dataset + 1LL*i * d), 4 * d);
  }
}

void read_fbin(ifstream& file, float*& dataset, int& n, int& d){
  file.read((char*)&n, 4);
  file.read((char*)&d, 4);
  printf("n = %d, d = %d\n", n, d);
  dataset = new float[1LL*n * d];
  file.read((char*)dataset, 1LL* 4 * n * d);
}

struct node{
  int id;
  float dis;
  bool operator<(const node& a) const{
    return dis < a.dis;
  }
};

__global__ void calc_dis(float* dataset, float* trainset, node* dis_idx, int n, int d, int t, int test_d){
  long long idx = 1LL*blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= 1LL*n*t) return;
  int i = idx / n;
  int j = idx % n;
  float dis = 0;
  for(int k = 0; k < test_d; k++){
    dis += (dataset[1LL*j * d + k] - trainset[1LL*i * d + k]) * (dataset[1LL*j * d + k] - trainset[1LL*i * d + k]);
  }
  dis_idx[idx].id = j;
  dis_idx[idx].dis = dis;
}

bool check(int test_d, float* dataset, float* trainset, int* traingt, int n, int d, int t, int gt_k, int k, int nc, float goal = 0.99){
  thrust::host_vector<node> h_dis_idx(1LL*n*t);
  thrust::device_vector<node> d_dis_idx(1LL*n*t);
  thrust::device_vector<float> d_dataset(dataset, dataset + 1LL*n * d);
  thrust::device_vector<float> d_trainset(trainset, trainset + 1LL*t * d);
  int threads_per_block = 256;
  int num_blocks = (1LL*n*t + threads_per_block - 1) / threads_per_block;
  calc_dis<<<num_blocks, threads_per_block>>>(d_dataset.data().get(), d_trainset.data().get(), d_dis_idx.data().get(), n, d, t, test_d);
  cudaDeviceSynchronize();
  thrust::copy(d_dis_idx.begin(), d_dis_idx.end(), h_dis_idx.begin());
  
  for(int i=0; i<t; i++){
    sort(h_dis_idx.begin() + 1LL*i*n, h_dis_idx.begin() + 1LL*(i+1)*n);
  }

  float recall = 0;
  for(int i=0; i<t; i++){
    for(int j=0; j<k; j++){
      int gt = traingt[1LL*i * gt_k + j];

      bool flag = 0;
      for(int l=0; l<nc; l++){
        if(h_dis_idx[1LL*i * n + l].id == gt){
          flag = 1;
          break;
        }
      }
      if(flag) recall += 1;
    }
  }
  recall = recall/(t*k);
  printf("test_d = %d, recall = %f\n", test_d, recall);
  return recall >= 0.99;
}

int main(){
  // ifstream dataset_file("/home/myl/pcsearch/bin/sift1M/pca_base.fbin", ios::binary);
  // ifstream trainset_file("/data/myl/sift1M/sift1M_trainset.fbin", ios::binary);
  // ifstream traingt_file("/data/myl/sift1M/sift1M_traingt.ivecs", ios::binary);

  // ifstream dataset_file("/home/myl/pcsearch/bin/sift10M/pca_base.fbin", ios::binary);
  // ifstream trainset_file("/data/myl/sift10M/sift10M_trainset.fbin", ios::binary);
  // ifstream traingt_file("/data/myl/sift10M/sift10M_traingt.ivecs", ios::binary);

  // ifstream dataset_file("/home/myl/pcsearch/bin/deep1M/pca_base.fbin", ios::binary);
  // ifstream trainset_file("/data/myl/deep1M/deep1M_trainset.fbin", ios::binary);
  // ifstream traingt_file("/data/myl/deep1M/deep1M_traingt.ivecs", ios::binary);

  // ifstream dataset_file("/home/myl/pcsearch/bin/gist/pca_base.fbin", ios::binary);
  // ifstream trainset_file("/data/myl/gist/gist_trainset.fbin", ios::binary);
  // ifstream traingt_file("/data/myl/gist/gist_traingt.ivecs", ios::binary);

  ifstream dataset_file("/home/myl/pcsearch/bin/COCO-I2I/pca_base.fbin", ios::binary);
  ifstream trainset_file("/data/myl/COCO-I2I/COCO-I2I_trainset.fbin", ios::binary);
  ifstream traingt_file("/data/myl/COCO-I2I/COCO-I2I_traingt.ivecs", ios::binary);

  float* dataset;
  float* trainset;
  int* traingt;
  int n, d;
  int t=100, k=10, gt_k;
  int nc;
  printf("input nc: ");
  scanf("%d", &nc);

  read_fbin(dataset_file, dataset, n, d);
  read_fbin(trainset_file, trainset, t, d);
  read_ivecs(traingt_file, traingt, t, gt_k);

  // check(39, dataset, trainset, traingt, n, d, t, gt_k, k, nc);
  int l=1, r=d;
  int ans = d;
  while(l<r){
    printf("%d %d\n", l, r);
    int mid = (l+r)/2;
    if(check(mid, dataset, trainset, traingt, n, d, t, gt_k, k, nc)){
      r = mid-1;
      ans = mid;
    }
    else l = mid+1;
  }
  printf("ans = %d\n", ans);
  return 0;
}