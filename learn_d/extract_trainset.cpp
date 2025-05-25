#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

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

int main(){
  // ifstream dataset_file("/home/myl/pcsearch/bin/sift1M/pca_base.fbin", ios::binary);
  // ofstream trainset_file("/data/myl/sift1M/sift1M_trainset.fbin", ios::binary);
  // ofstream traingt_file("/data/myl/sift1M/sift1M_traingt.ivecs", ios::binary);

  // ifstream dataset_file("/home/myl/pcsearch/bin/sift10M/pca_base.fbin", ios::binary);
  // ofstream trainset_file("/data/myl/sift10M/sift10M_trainset.fbin", ios::binary);
  // ofstream traingt_file("/data/myl/sift10M/sift10M_traingt.ivecs", ios::binary);

  // ifstream dataset_file("/home/myl/pcsearch/bin/deep1M/pca_base.fbin", ios::binary);
  // ofstream trainset_file("/data/myl/deep1M/deep1M_trainset.fbin", ios::binary);
  // ofstream traingt_file("/data/myl/deep1M/deep1M_traingt.ivecs", ios::binary);

  // ifstream dataset_file("/home/myl/pcsearch/bin/gist/pca_base.fbin", ios::binary);
  // ofstream trainset_file("/data/myl/gist/gist_trainset.fbin", ios::binary);
  // ofstream traingt_file("/data/myl/gist/gist_traingt.ivecs", ios::binary);

  ifstream dataset_file("/home/myl/pcsearch/bin/COCO-I2I/pca_base.fbin", ios::binary);
  ofstream trainset_file("/data/myl/COCO-I2I/COCO-I2I_trainset.fbin", ios::binary);
  ofstream traingt_file("/data/myl/COCO-I2I/COCO-I2I_traingt.ivecs", ios::binary);
  
  float* dataset;
  int n, d;
  int t = 100;
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
      trainset[i*d+j] = dataset[1LL*idx[i]*d+j];
    }
  }
  for(int i=0; i<10; i++) printf("%f ", trainset[i]);printf("\n");
  trainset_file.write((char*)&t, 4);
  trainset_file.write((char*)&d, 4);
  trainset_file.write((char*)trainset, 4 * t * d);

  //生成traingt
  printf("calc ground truth\n");
  pair<float, int>* dis_idx = new pair<float, int>[n];
  for(int i=0; i<t; i++){
    for(int j=0; j<n; j++){
      float dis = 0;
      for(int l=0; l<d; l++){
        dis = dis + (trainset[i*d+l] - dataset[j*d+l]) * (trainset[i*d+l] - dataset[j*d+l]);
      }
      dis_idx[j] = make_pair(dis, j);
    }
    sort(dis_idx, dis_idx + n);
    traingt_file.write((char*)&k, 4);
    for(int j=0; j<k; j++){
      traingt_file.write((char*)&dis_idx[j].second, 4);
    }
  }


  dataset_file.close();
  trainset_file.close();
  traingt_file.close();
  return 0;
}