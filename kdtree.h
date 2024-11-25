#pragma once
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <optix_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class Kdtree{
private:
  // struct Kdnode{
  //   uint l, r;
  //   OptixAabb box_bound;
  // };
public:
  Kdtree(){}
  Kdtree(float3* _data, uint _n, uint _max_node, OptixAabb world_bound, thrust::host_vector<OptixAabb> &aabbs);
  ~Kdtree();
  void build(uint l, uint r, uint node_id, OptixAabb box_bound, thrust::host_vector<OptixAabb> &aabbs);
  uint split(uint l, uint r, uint axis, float xM);
  void add_aabb(uint l, uint r, OptixAabb box_bound, thrust::host_vector<OptixAabb> &aabbs);
  void computeAabbPid(thrust::device_vector<uint> &aabb_pid, thrust::device_vector<uint> &prefix_sum, uint n_aabbs);

  float** data;
  uint* id;
  uint n;
  uint max_node;
  uint* belong;
};