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
  #ifdef DETAIL
    printf("Read data function\n");
  #endif
  if(std::string(get_last_m_chars(datafile, 3)) == "txt"){
      #ifdef DETAIL
        printf("Reading a txt file\n");
      #endif
      read_txt_file(datafile, n, d, data);
      #ifdef DETAIL
        printf("n = %d, d = %d\n", n, d);
      #endif
    }
    else if(std::string(get_last_m_chars(datafile, 5)) == "fvecs"){
      #ifdef DETAIL
        printf("Reading a fvecs file\n");
      #endif
      read_fvecs_file(datafile, n, d, data);
      #ifdef DETAIL
        printf("n = %d, d = %d\n", n, d);
      #endif
    }
    else if(std::string(get_last_m_chars(datafile, 4)) == "fbin"){
      #ifdef DETAIL
        printf("Reading a fbin file\n");
      #endif
      read_fbin_file(datafile, n, d, data);
      #ifdef DETAIL
        printf("n = %d, d = %d\n", n, d);
      #endif
    }
    else if(std::string(get_last_m_chars(datafile, 5)) == "hvecs"){
      #ifdef DETAIL
        printf("Reading a hvecs file\n");
      #endif
      read_hvecs_file(datafile, n, d, data);
      #ifdef DETAIL
        printf("n = %d, d = %d\n", n, d);
      #endif
    }
    else if(std::string(get_last_m_chars(datafile, 5)) == "bvecs"){
      #ifdef DETAIL
        printf("Reading a bvecs file\n");
      #endif
      read_bvecs_file(datafile, n, d, data);
      #ifdef DETAIL
        printf("n = %d, d = %d\n", n, d);
      #endif
    }
    else if(std::string(get_last_m_chars(datafile, 2)) == "hh"){
      #ifdef DETAIL
        printf("Reading a hh file\n");
      #endif
      read_hh_file(datafile, n, d, data);
      #ifdef DETAIL
        printf("n = %d, d = %d\n", n, d);
      #endif
    }
    else{
      #ifdef DETAIL
        printf("Unknown file type\n");
      #endif
    }
}

void file_read::read_hh_file(const char* filename, uint& n, uint& d,
                             thrust::host_vector<float>& data){
  // FILE* file = fopen(filename, "rb");
  // if(file == NULL){
  //   printf("File open failed\n");
  //   return;
  // }
  // assert(fread(&d, sizeof(d), 1, file) == 1);
  // long long filelength = 0;
  // fseek(file, 0, SEEK_END);
  // filelength = ftell(file);
  // fseek(file, 0, SEEK_SET);
  // n = (filelength - 4) / (d*4);
  // size_t dataSize = size_t(n) * size_t(d);
  // data.resize(dataSize);
  // fread(&d, sizeof(d), 1, file);
  // fread(data.data(), sizeof(float), 1UL*dataSize, file);
  // fclose(file);
  std::ifstream infile(filename, std::ios::binary);
  if(!infile.is_open()){
    printf("File open failed\n");
    exit(1);
  }
  infile.read((char*)&d, 4);

  long long filelength;
  infile.seekg(0, std::ios::end);
  filelength = infile.tellg();
  n = (filelength - 4) / (d*4);

  infile.seekg(4, std::ios::beg);
  data.resize(n*d);
  auto* data_ptr = thrust::raw_pointer_cast(data.data());
  infile.read((char*)data_ptr, 1LL*n*d*sizeof(float));
  infile.close();
}

void file_read::read_txt_file(const char *filename, uint &n, uint &d,
                              thrust::host_vector<float> &data) {
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
  // FILE *file = fopen(filename, "rb");
  // if (file == NULL) {
  //   printf("File open failed\n");
  //   return;
  // }

  // int rtn = fread(&d, sizeof(d), 1, file);
  // assert(rtn == 1);
  // long long filelength = 0;
  // fseek(file, 0, SEEK_END);
  // filelength = ftell(file);
  // fseek(file, 0, SEEK_SET);
  // n = filelength / ((d + 1) * 4);

  // data.resize(n * d);
  // for (int i = 0; i < n; i++) {
  //   fread(&d, sizeof(d), 1, file);
  //   fread(data.data() + i * d, 4, d, file);
  // }
  // fclose(file);

  std::ifstream infile(filename, std::ios::binary);
  if(!infile.is_open()){
    printf("File open failed\n");
    exit(1);
  }
  infile.read((char*)&d, 4);
  long long filelength;
  infile.seekg(0, std::ios::end);
  filelength = infile.tellg();
  n = filelength / ((d+1)*4);

  data.resize(n*d);
  infile.seekg(0, std::ios::beg);
  auto* data_ptr = thrust::raw_pointer_cast(data.data());
  for(int i=0; i<n; i++){
    infile.seekg(4, std::ios::cur);
    infile.read((char*)(data_ptr+i*d), 4*d);
  }
  infile.close();
}

void file_read::read_fbin_file(const char *filename, uint &n, uint &d,
                               thrust::host_vector<float> &data) {
  // FILE *file = fopen(filename, "rb");
  // if (file == NULL) {
  //   printf("File open failed\n");
  //   return;
  // }
  // assert(fread(&n, sizeof(n), 1, file));
  // // uint limit = 10000000; // 最多读取1M
  // // n = std::min(n, limit);
  // assert(fread(&d, sizeof(d), 1, file) == 1);
  // data.resize(n * d);
  // fread(data.data(), sizeof(float), 1LL * size_t(n) * size_t(d), file);
  // fclose(file);

  std::ifstream infile(filename, std::ios::binary);
  if(!infile.is_open()){
    printf("File open failed\n");
    exit(1);
  }
  infile.read((char*)&n, 4);
  infile.read((char*)&d, 4);
  data.resize(1LL*n*d);
  auto* data_ptr = thrust::raw_pointer_cast(data.data());
  infile.read((char*)data_ptr, 1LL*n*d*sizeof(float));
  infile.close();
}

void file_read::read_hvecs_file(const char* filename, uint& n, uint& d,
                                thrust::host_vector<float>& data){
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
  
  data.resize(n*d);
  fread(&d, sizeof(d), 1, file);
  auto* data_ptr = thrust::raw_pointer_cast(data.data());
  fread(data_ptr, sizeof(float), n*d, file);
  fclose(file);
}

void file_read::read_bvecs_file(const char* filename, uint& n, uint& d,
                                thrust::host_vector<float>& data){
  // FILE* file = fopen(filename, "rb");
  // if(file == NULL){
  //   printf("File open failed\n");
  //   return;
  // }
  // assert(fread(&d, sizeof(d), 1, file) == 1);
  // int filelength = 0;
  // fseek(file, 0, SEEK_END);
  // filelength = ftell(file);
  // fseek(file, 0, SEEK_SET);
  // n = filelength / (d+4);
  
  // data.resize(n*d);
  // for(int i = 0; i < n; i++){
  //   fread(&d, sizeof(d), 1, file);
  //   unsigned char tmp_data[d];
  //   fread(tmp_data, 1, d, file);
  //   for(int j=0; j<d; j++){
  //     data[i*d+j] = (float)tmp_data[j];
  //   }
  // }
  // fclose(file);

  std::ifstream infile(filename, std::ios::binary);
  if(!infile.is_open()){
    printf("File open failed\n");
    exit(1);
  }
  infile.read((char*)&d, 4);
  long long filelength;
  infile.seekg(0, std::ios::end);
  filelength = infile.tellg();
  n = filelength / (d+4);
  if(n>100000000) n = 100000000;//读取前100M

  data.resize(1LL * n*d);
  infile.seekg(0, std::ios::beg);
  auto* data_ptr = thrust::raw_pointer_cast(data.data());
  for(int i=0; i<n; i++){
    infile.seekg(4, std::ios::cur);
    unsigned char tmp_data[d];
    infile.read((char*)tmp_data, d);
    for(int j=0; j<d; j++){
      data_ptr[1LL*i*d+j] = (float)tmp_data[j];
    }
  }
  infile.close();
}

void file_read::read_ivecs_file(const char* filename, uint& n, uint& d,
                                thrust::host_vector<uint>& data){
  // FILE* file = fopen(filename, "rb");
  // if(file == NULL){
  //   printf("File open failed\n");
  //   return;
  // }
  // assert(fread(&d, sizeof(d), 1, file) == 1);
  // long long filelength = 0;
  // fseek(file, 0, SEEK_END);
  // filelength = ftell(file);
  // fseek(file, 0, SEEK_SET);
  // n = filelength / ((d+1)*4);
  
  // data.resize(n*d);
  // for(int i = 0; i < n; i++){
  //   fread(&d, sizeof(d), 1, file);
  //   fread(data.data()+i*d, sizeof(uint), d, file);
  // }
  // fclose(file);

  std::ifstream infile(filename, std::ios::binary);
  if(!infile.is_open()){
    printf("File open failed\n");
    exit(1);
  }
  infile.read((char*)&d, 4);
  long long filelength;
  infile.seekg(0, std::ios::end);
  filelength = infile.tellg();
  n = filelength / ((d+1)*4);

  data.resize(n*d);
  infile.seekg(0, std::ios::beg);
  auto* data_ptr = thrust::raw_pointer_cast(data.data());
  for(int i=0; i<n; i++){
    infile.seekg(4, std::ios::cur);
    infile.read((char*)(data_ptr+i*d), 4*d);
  }
  infile.close();
}

void file_read::read_graph(const char* filename, const uint& n, uint& degree, thrust::host_vector<uint> &data){
  // FILE* file = fopen(filename, "rb");
  // if(file == NULL){
  //   printf("File open failed\n");
  //   return;
  // }
  // long long filelength = 0;
  // fseek(file, 0, SEEK_END);
  // filelength = ftell(file);
  // fseek(file, 0, SEEK_SET);
  // degree = filelength / (n*4);

  // size_t dataSize = size_t(n) * size_t(degree);
  // data.resize(dataSize);
  // fread(data.data(), sizeof(uint), 1UL*dataSize, file);
  // fclose(file);

  std::ifstream infile(filename, std::ios::binary);
  if(!infile.is_open()){
    printf("File open failed\n");
    exit(1);
  }
  long long filelength;
  infile.seekg(0, std::ios::end);
  filelength = infile.tellg();
  infile.seekg(0, std::ios::beg);
  degree = filelength / (1LL*n*4);

  data.resize(1LL*n*degree);
  auto* data_ptr = thrust::raw_pointer_cast(data.data());
  infile.read((char*)data_ptr, 1LL*n*degree*sizeof(uint));
  infile.close();
}