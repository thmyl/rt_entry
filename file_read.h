#pragma once
#include <cstdlib>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <fstream>
#include <vector>

struct ClusterData;

class file_read{
  public:
    static void read_data(const char*, int&, int&, thrust::host_vector<float> &data);
    static void read_txt_file(const char*, int&, int&, thrust::host_vector<float> &data);
    static void read_fvecs_file(const char*, int&, int&, thrust::host_vector<float> &data);
    static void read_fbin_file(const char*, int&, int&, thrust::host_vector<float> &data);
    static void read_hvecs_file(const char*, int&, int&, thrust::host_vector<float> &data);
    static void read_bvecs_file(const char*, int&, int&, thrust::host_vector<float> &data);
    static void read_ivecs_file(const char*, int&, int&, thrust::host_vector<int> &data);
    static void read_graph(const char*, const int&, int&, thrust::host_vector<int> &data);
    static void read_hh_file(const char*, int&, int&, thrust::host_vector<float> &data);
    static void read_centroids(const char*, ClusterData& cluster_data, int np, int dim);
    file_read(){};
};