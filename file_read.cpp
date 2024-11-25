#include "file_read.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

static const char *get_last_m_chars(const char *str, int m) {
  int len = std::strlen(str); // 获取字符串的长度
  if (len <= m) {
    return str; // 如果待获取的后m位长度大于字符串长度，则返回整个字符串
  } else {
    return str + len - m; // 返回后m位的起始位置
  }
}

void file_read::read_data(const char *datafile, uint &n, uint &d,
                          thrust::host_vector<float> &data){
  printf("Read data function\n");
  if(std::string(get_last_m_chars(datafile, 3)) == "txt"){
      printf("Reading a txt file\n");
      read_txt_file(datafile, n, d, data);
      printf("n = %d, d = %d\n", n, d);
    }
    else if(std::string(get_last_m_chars(datafile, 5)) == "fvecs"){
      printf("Reading a fvecs file\n");
      read_fvecs_file(datafile, n, d, data);
      printf("n = %d, d = %d\n", n, d);
    }
    else if(std::string(get_last_m_chars(datafile, 4)) == "fbin"){
      printf("Reading a fbin file\n");
      read_fbin_file(datafile, n, d, data);
      printf("n = %d, d = %d\n", n, d);
    }
    else if(std::string(get_last_m_chars(datafile, 5)) == "hvecs"){
      printf("Reading a fvecs file\n");
      read_hvecs_file(datafile, n, d, data);
      printf("n = %d, d = %d\n", n, d);
    }
    else if(std::string(get_last_m_chars(datafile, 5)) == "bvecs"){
      printf("Reading a fvecs file\n");
      read_bvecs_file(datafile, n, d, data);
      printf("n = %d, d = %d\n", n, d);
    }
}

void file_read::read_txt_file(const char *filename, uint &n, uint &d,
                              thrust::host_vector<float> &data) {
  printf("Read txt file function\n");
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    printf("File open failed\n");
    return;
  }
  fscanf(file, "%d %d", &n, &d);
  data.resize(n * d);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d - 1; j++) {
      fscanf(file, "%f,", &data[i * d + j]);
    }
    fscanf(file, "%f", &data[i * d + d - 1]);
  }
  fclose(file);
}

void file_read::read_fvecs_file(const char *filename, uint &n, uint &d,
                                thrust::host_vector<float> &data) {
  printf("Read fvecs file function\n");
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    printf("File open failed\n");
    return;
  }

  int rtn = fread(&d, sizeof(d), 1, file);
  assert(rtn == 1);
  int filelength = 0;
  fseek(file, 0, SEEK_END);
  filelength = ftell(file);
  fseek(file, 0, SEEK_SET);
  n = filelength / ((d + 1) * 4);
  printf("n: %d, d: %d\n", n, d);

  data.resize(n * d);
  for (int i = 0; i < n; i++) {
    fread(&d, sizeof(d), 1, file);
    fread(data.data() + i * d, 4, d, file);
  }
  fclose(file);
}

void file_read::read_fbin_file(const char *filename, uint &n, uint &d,
                               thrust::host_vector<float> &data) {
  printf("Read fin file function\n");
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    printf("File open failed\n");
    return;
  }
  auto rtn = fread(&n, sizeof(n), 1, file);
  assert(rtn == 1);
  uint limit = 10000000; // 最多读取1M
  n = std::min(n, limit);
  assert(fread(&d, sizeof(d), 1, file) == 1);
  data.resize(n * d);
  fread(data.data(), sizeof(float), n * d, file);
  fclose(file);
}

void file_read::read_hvecs_file(const char* filename, uint& n, uint& d,
                                thrust::host_vector<float>& data){
  printf("Read hvecs file function\n");
  FILE* file = fopen(filename, "rb");
  if(file == NULL){
    printf("File open failed\n");
    return;
  }
  assert(fread(&d, sizeof(d), 1, file) == 1);
  int filelength = 0;
  fseek(file, 0, SEEK_END);
  filelength = ftell(file);
  fseek(file, 0, SEEK_SET);
  n = (filelength / 4 - 1)/d;
  printf("n: %d, d: %d\n", n, d);
  
  data.resize(n*d);
  fread(&d, sizeof(d), 1, file);
  fread(data.data(), sizeof(float), n*d, file);
  fclose(file);
}

void file_read::read_bvecs_file(const char* filename, uint& n, uint& d,
                                thrust::host_vector<float>& data){
  printf("Read bvecs file function\n");
  FILE* file = fopen(filename, "rb");
  if(file == NULL){
    printf("File open failed\n");
    return;
  }
  assert(fread(&d, sizeof(d), 1, file) == 1);
  int filelength = 0;
  fseek(file, 0, SEEK_END);
  filelength = ftell(file);
  fseek(file, 0, SEEK_SET);
  n = filelength / (d+4);
  printf("n: %d, d: %d\n", n, d);
  
  data.resize(n*d);
  for(int i = 0; i < n; i++){
    fread(&d, sizeof(d), 1, file);
    unsigned char tmp_data[d];
    fread(tmp_data, 1, d, file);
    for(int j=0; j<d; j++){
      data[i*d+j] = (float)tmp_data[j];
    }
  }
  fclose(file);
}

void file_read::read_ivecs_file(const char* filename, uint& n, uint& d,
                                thrust::host_vector<uint>& data){
  printf("Read ivecs file function\n");
  FILE* file = fopen(filename, "rb");
  if(file == NULL){
    printf("File open failed\n");
    return;
  }
  assert(fread(&d, sizeof(d), 1, file) == 1);
  int filelength = 0;
  fseek(file, 0, SEEK_END);
  filelength = ftell(file);
  fseek(file, 0, SEEK_SET);
  n = filelength / ((d+1)*4);
  printf("n: %d, d: %d\n", n, d);
  
  data.resize(n*d);
  for(int i = 0; i < n; i++){
    fread(&d, sizeof(d), 1, file);
    fread(data.data()+i*d, sizeof(uint), d, file);
  }
  fclose(file);
}