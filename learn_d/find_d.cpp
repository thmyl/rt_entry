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
#pragma omp declare reduction(vecmin: int: omp_out = std::min(omp_out, omp_in)) initializer(omp_priv = INT_MAX)

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

int main(){
  // ifstream dataset_file("/home/myl/pcsearch/bin/sift1M/pca_base.fbin", ios::binary);
  // ifstream trainset_file("/data/myl/sift1M/sift1M_trainset.fbin", ios::binary);
  // ifstream traingt_file("/data/myl/sift1M/sift1M_traingt.ivecs", ios::binary);
  // ofstream outfile("./sift1M_d.txt");
  // int nc = 512;

  ifstream dataset_file("/home/myl/pcsearch/bin/deep1M/pca_base.fbin", ios::binary);
  ifstream trainset_file("/data/myl/deep1M/deep1M_trainset.fbin", ios::binary);
  ifstream traingt_file("/data/myl/deep1M/deep1M_traingt.ivecs", ios::binary);
  ofstream outfile("/data/myl/deep1M/deep1M_d.txt");
  int nc = 512;

  // ifstream dataset_file("/home/myl/pcsearch/bin/gist/pca_base.fbin", ios::binary);
  // ifstream trainset_file("/data/myl/gist/gist_trainset.fbin", ios::binary);
  // ifstream traingt_file("/data/myl/gist/gist_traingt.ivecs", ios::binary);
  // ofstream outfile("./gist_d.txt");
  // int nc = 512;

  float* dataset;
  float* trainset;
  int* traingt;
  int n, d;
  int t=4616, k=10, gt_k;
  double tau = 0.99;
  // int nc;
  // printf("input nc: ");
  // scanf("%d", &nc);

  read_fbin(dataset_file, dataset, n, d);
  read_fbin(trainset_file, trainset, t, d);
  read_ivecs(traingt_file, traingt, t, gt_k);

  pair<float, int>* dis_idx = new pair<float, int>[1LL*n*t];
  for(long long i=0; i<1LL*n*t; i++){
    dis_idx[i].second = i%n;
    dis_idx[i].first = 0;
  }
  // for(int i=0; i<d; i++){
  //   printf("Processing dimension %d...\n", i+1);
  //   double recall = 0;
  //   for(int train_id = 0; train_id < t; train_id++){
  //     for(int j=0; j<n; j++){
  //       int point_id = dis_idx[1LL*train_id*n + j].second;
  //       dis_idx[1LL*train_id*n + j].first += (dataset[1LL* point_id * d + i] - trainset[1LL * train_id * d + i]) * (dataset[1LL* point_id * d + i] - trainset[1LL*train_id * d + i]);
  //     }
  //     sort(dis_idx + 1LL*train_id*n, dis_idx + 1LL*(train_id+1)*n);
  //     for(int j=0; j<k; j++){
  //       int gt = traingt[1LL*train_id * gt_k + j];
  //       for(int l=0; l<nc; l++){
  //         if(dis_idx[1LL*train_id*n + l].second == gt){
  //           recall = recall+1;
  //           break;
  //         }
  //       }
  //     }
  //   }
  //   recall = recall / (t * k);
  //   outfile << i+1 << " " << recall << std::endl;
  //   if(recall >= tau){
  //     printf("d = %d\n", i+1);
  //     break;
  //   }
  // }

  for(int i = 0; i < d; i++) {
      printf("Processing dimension %d...\n", i + 1);
      double recall = 0;

      // 并行 train_id 层
      #pragma omp parallel for reduction(+:recall)
      for(int train_id = 0; train_id < t; train_id++) {
          for(int j = 0; j < n; j++) {
              int point_id = dis_idx[1LL * train_id * n + j].second;
              dis_idx[1LL * train_id * n + j].first +=
                  (dataset[1LL * point_id * d + i] - trainset[1LL * train_id * d + i]) *
                  (dataset[1LL * point_id * d + i] - trainset[1LL * train_id * d + i]);
          }
          sort(dis_idx + 1LL*train_id*n, dis_idx + 1LL*(train_id+1)*n);
          for(int j = 0; j < k; j++) {
              int gt = traingt[1LL * train_id * gt_k + j];
              for(int l = 0; l < nc; l++) {
                  if(dis_idx[1LL*train_id*n + l].second == gt) {
                      recall += 1;
                      break;
                  }
              }
          }
      }

      recall = recall / (t * k);
      outfile << i + 1 << " " << recall << std::endl;

      if(recall >= tau) {
          printf("d = %d\n", i + 1);
          break;
      }
  }
  return 0;
}