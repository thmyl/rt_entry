#pragma once
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <optix_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stack>

class Kdtree{
private:
  // struct Kdnode{
  //   int l, r;
  //   OptixAabb box_bound;
  // };
public:
  Kdtree(){}
  Kdtree(float3* _data, int _n, int _max_node, OptixAabb world_bound, thrust::host_vector<OptixAabb> &aabbs);
  ~Kdtree();
  void build(int l, int r, int node_id, OptixAabb box_bound, thrust::host_vector<OptixAabb> &aabbs);
  void buildWithStack(int l, int r, int node_id, OptixAabb box_bound, thrust::host_vector<OptixAabb> &aabbs);
  int split(int l, int r, int axis, float xM);
  void add_aabb(int l, int r, OptixAabb box_bound, thrust::host_vector<OptixAabb> &aabbs);
  // void computeAabbPid(thrust::device_vector<int> &aabb_pid, thrust::device_vector<int> &prefix_sum, int n_aabbs);
  void computeAabbPid(thrust::device_vector<int> &aabb_entry, int n_aabbs);
  float findxM(float l, float r, int data_l, int data_r, int axis);
  void tight_box(int l, int r, float *box_min, float *box_max);

  std::vector<std::vector<float>> data;
  int* id;
  int n;
  int max_node;
  int* belong;
  const float eps = 1e-6;
};