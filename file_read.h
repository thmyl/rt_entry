#pragma once
#include <cstdlib>
#include <thrust/host_vector.h>

class file_read{
  public:
    static void read_data(const char*, uint&, uint&, thrust::host_vector<float> &data);
    static void read_txt_file(const char*, uint&, uint&, thrust::host_vector<float> &data);
    static void read_fvecs_file(const char*, uint&, uint&, thrust::host_vector<float> &data);
    static void read_fbin_file(const char*, uint&, uint&, thrust::host_vector<float> &data);
    static void read_hvecs_file(const char*, uint&, uint&, thrust::host_vector<float> &data);
    static void read_bvecs_file(const char*, uint&, uint&, thrust::host_vector<float> &data);
    static void read_ivecs_file(const char*, uint&, uint&, thrust::host_vector<uint> &data);
    static void read_graph(const char*, const uint&, uint&, thrust::host_vector<uint> &data);
    file_read(){};
};