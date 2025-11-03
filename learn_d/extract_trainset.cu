#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
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

void read_fbin(ifstream& file, float*& dataset, int& n, int& d){
  file.read((char*)&n, 4);
  file.read((char*)&d, 4);
  printf("n = %d, d = %d\n", n, d);
  dataset = new float[1LL*n * d];
  file.read((char*)dataset, 1LL* 4 * n * d);
}

__global__ void distance_kernel(float* d_dataset, float* d_trainset, pair<float, int>* d_dis_idx, int n, int t, int d){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid >= n) return;
  int point_id = tid;
  for(int i=0; i<t; i++){
    int id = 1LL*i * n + point_id;
    d_dis_idx[id].second = point_id;
    d_dis_idx[id].first = 0;
    for(int j=0; j<d; j++){
      d_dis_idx[id].first += (d_trainset[1LL*i*d+j] - d_dataset[1LL*point_id*d+j]) * (d_trainset[1LL*i*d+j] - d_dataset[1LL*point_id*d+j]);
    }
  }
}

void SetDevice(int device_id=0){
    int device_count=0;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop,device_id);
    #ifdef DETAIL
      printf("Maximum dimensions of grid size: (%d, %d, %d)\n",
            device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
      printf("Maximum dimensions of block size: (%d, %d, %d)\n",
            device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
    #endif
    size_t available_memory,total_memory;
    cudaMemGetInfo(&available_memory,&total_memory);
    // std::cout<<"==========================================================\n";
    std::cout<<"Total GPUs visible: "<<device_count;
    std::cout<<", using ["<<device_id<<"]: "<<device_prop.name<<std::endl;
    std::cout<<"Available Memory: "<<int(available_memory/1024/1024)<<" MB, ";
    std::cout<<"Total Memory: "<<int(total_memory/1024/1024)<<" MB\n";
}

int main(){
  // ifstream dataset_file("/home/myl/pcsearch/bin/sift1M/pca_base.fbin", ios::binary);
  // ofstream trainset_file("/data/myl/sift1M/sift1M_trainset.fbin", ios::binary);
  // ofstream traingt_file("/data/myl/sift1M/sift1M_traingt.ivecs", ios::binary);
  // SetDevice(0);

  // ifstream dataset_file("/home/myl/pcsearch/bin/deep1M/pca_base.fbin", ios::binary);
  // ofstream trainset_file("/data/myl/deep1M/deep1M_trainset.fbin", ios::binary);
  // ofstream traingt_file("/data/myl/deep1M/deep1M_traingt.ivecs", ios::binary);
  // SetDevice(1);

  ifstream dataset_file("/home/myl/pcsearch/bin/gist/pca_base.fbin", ios::binary);
  ofstream trainset_file("/data/myl/gist/gist_trainset.fbin", ios::binary);
  ofstream traingt_file("/data/myl/gist/gist_traingt.ivecs", ios::binary);
  SetDevice(2);
  
  float* dataset;
  int n, d;
  int t = 4616;
  int k = 100;
  read_fbin(dataset_file, dataset, n, d);

  //生成0~n-1的随机序列
  int* idx = new int[n];
  for(int i = 0; i < n; i++) idx[i] = i;
  srand(43);
  random_shuffle(idx, idx + n);
  for(int i = 0; i < t; i++) printf("%d ", idx[i]);
  printf("\n");

  //生成训练集
  float* trainset = new float[1LL*t * d];
  for(int i = 0; i < t; i++){
    for(int j=0; j<d; j++){
      trainset[1LL*i*d+j] = dataset[1LL*idx[i]*d+j];
    }
  }
  for(int i=0; i<10; i++) printf("%f ", trainset[i]);printf("\n");
  trainset_file.write((char*)&t, 4);
  trainset_file.write((char*)&d, 4);
  trainset_file.write((char*)trainset, 4 * t * d);

  //生成traingt
  printf("calc ground truth\n");
  // pair<float, int>* dis_idx = new pair<float, int>[n];
  // for(int i=0; i<t; i++){
  //   for(int j=0; j<n; j++){
  //     float dis = 0;
  //     for(int l=0; l<d; l++){
  //       dis = dis + (trainset[i*d+l] - dataset[j*d+l]) * (trainset[i*d+l] - dataset[j*d+l]);
  //     }
  //     dis_idx[j] = make_pair(dis, j);
  //   }
  //   sort(dis_idx, dis_idx + n);
  //   traingt_file.write((char*)&k, 4);
  //   for(int j=0; j<k; j++){
  //     traingt_file.write((char*)&dis_idx[j].second, 4);
  //   }
  // }

  float* d_dataset, * d_trainset;
  pair<float, int>* d_dis_idx;
  int batch_size = 512;
  cudaMalloc((void**)&d_dataset, 1LL*n*d*sizeof(float));
  cudaMalloc((void**)&d_trainset, 1LL*t*d*sizeof(float));
  cudaMalloc((void**)&d_dis_idx, 1LL*batch_size*n*sizeof(pair<float, int>));
  cudaMemcpy(d_dataset, dataset, 1LL*n*d*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_trainset, trainset, 1LL*t*d*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_dis_idx, 0, 1LL*batch_size*n*sizeof(pair<float, int>));
  pair<float, int>* dis_idx = new pair<float, int>[1LL*t*n];

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  for(int batch_id = 0; batch_id<(t+batch_size-1)/batch_size; batch_id++){
    printf("batch %d\n", batch_id);
    int start = 1LL*batch_id * batch_size;
    int end = min((batch_id + 1) * batch_size, t);
    int current_batch_size = end - start;
    cudaMemset(d_dis_idx, 0, 1LL*current_batch_size*n*sizeof(pair<float, int>));
    distance_kernel<<<gridSize, blockSize>>>(d_dataset, d_trainset + 1LL*start*d, d_dis_idx, n, current_batch_size, d);
    cudaDeviceSynchronize();
    cudaMemcpy(dis_idx + 1LL*start*n, d_dis_idx, 1LL*current_batch_size*n*sizeof(pair<float, int>), cudaMemcpyDeviceToHost);
  }
  // distance_kernel<<<gridSize, blockSize>>>(d_dataset, d_trainset, d_dis_idx, n, t, d);
  // cudaDeviceSynchronize();
  // cudaMemcpy(dis_idx, d_dis_idx, 1LL*t*n*sizeof(pair<float, int>), cudaMemcpyDeviceToHost);
  for(int i=0; i<t; i++){
    sort(dis_idx+1LL*i*n, dis_idx+1LL*(i+1)*n);
    traingt_file.write((char*)&k, 4);
    for(int j=0; j<k; j++){
      traingt_file.write((char*)&dis_idx[1LL*i*n+j].second, 4);
    }
  }

  dataset_file.close();
  trainset_file.close();
  traingt_file.close();
  delete[] dataset;
  delete[] trainset;
  delete[] idx;
  delete[] dis_idx;
  cudaFree(d_dataset);
  cudaFree(d_trainset);
  cudaFree(d_dis_idx);
  printf("Training set and ground truth generated successfully.\n");
  return 0;
}