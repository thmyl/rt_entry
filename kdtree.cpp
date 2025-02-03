#include "kdtree.h"

Kdtree::Kdtree(float3* _data, uint _n, uint _max_node, OptixAabb world_bound, thrust::host_vector<OptixAabb> &aabbs){
  max_node = _max_node;
  n = _n;
  belong = new uint[n];
  data = new float*[n];
  for(uint i=0; i<n; i++){
    data[i] = new float[3];
    data[i][0] = _data[i].x;
    data[i][1] = _data[i].y;
    data[i][2] = _data[i].z;
  }
  id = new uint[n];
  for(int i=0; i<n; i++)id[i] = i;
  // printf("begin build...\n");
  // root = new Kdnode;
  build(0, n, 0, world_bound, aabbs);
}

Kdtree::~Kdtree(){
  for(uint i=0; i<n; i++)delete[] data[i];
  delete[] data;
  delete[] id;
  delete[] belong;
}

uint findWidest(OptixAabb box_bound){
  float max_width = 0;
  uint axis = 0;
  for(int dim=0; dim<3; dim++){
    float Min = reinterpret_cast<float*>(&box_bound.minX)[dim];
    float Max = reinterpret_cast<float*>(&box_bound.maxX)[dim];
    if(Max-Min>max_width){
      max_width = Max-Min;
      axis = dim;
    }
  }
  return axis;
}

uint Kdtree::split(uint l, uint r, uint axis, float xM){
  r = r-1;
  while(l < r){
    if(data[l][axis] >= xM){
      while(data[r][axis] >= xM && l < r)r--;
      if(l < r){
        for(int i=0; i<3; i++)
          std::swap(data[l][i], data[r][i]);
        std::swap(id[l], id[r]);
        r--;
      }
      else break;
    }
    l++;
  }
  if(data[l][axis] < xM)l++;
  return l;
}

void Kdtree::add_aabb(uint l, uint r, OptixAabb box_bound, thrust::host_vector<OptixAabb> &aabbs){
  float *box_min = reinterpret_cast<float*>(&box_bound.minX);
  float *box_max = reinterpret_cast<float*>(&box_bound.maxX);
  for(int i=0; i<3; i++){
    reinterpret_cast<float*>(&box_bound.minX)[i] = data[l][i];
    reinterpret_cast<float*>(&box_bound.maxX)[i] = data[l][i];
  }
  for(uint i=l+1; i<r; i++){
    for(int j=0; j<3; j++){
      reinterpret_cast<float*>(&box_bound.minX)[j] = std::min(reinterpret_cast<float*>(&box_bound.minX)[j], data[i][j]);
      reinterpret_cast<float*>(&box_bound.maxX)[j] = std::max(reinterpret_cast<float*>(&box_bound.maxX)[j], data[i][j]);
    }
  }
  uint aabb_id = aabbs.size();
  for(uint i=l; i<r; i++){
    belong[id[i]] = aabb_id;
  }
  aabbs.push_back(box_bound);
}

void Kdtree::computeAabbPid(thrust::device_vector<uint> &aabb_pid, thrust::device_vector<uint> &prefix_sum, uint n_aabbs){
  thrust::host_vector<uint> h_aabb_pid(n);
  thrust::host_vector<uint> h_prefix_sum(n_aabbs + 1);
  uint* count = new uint[n_aabbs];
  for(uint i=0; i<n_aabbs; i++)count[i] = 0;
  for(uint i=0; i<n; i++)count[belong[i]]++;
  h_prefix_sum[0] = 0;
  for(uint i=0; i<n_aabbs; i++){
    h_prefix_sum[i+1] = h_prefix_sum[i] + count[i];
  }
  delete[] count;
  thrust::copy(h_prefix_sum.begin(), h_prefix_sum.end(), prefix_sum.begin());
  for(uint i=0; i<n; i++){
    h_aabb_pid[h_prefix_sum[belong[i]]] = i;
    h_prefix_sum[belong[i]]++;
  }
  thrust::copy(h_aabb_pid.begin(), h_aabb_pid.end(), aabb_pid.begin());
}

void Kdtree::build(uint l, uint r, uint node_id, OptixAabb box_bound, thrust::host_vector<OptixAabb> &aabbs){
  if(r-l<=max_node){
    if(r>l) add_aabb(l, r, box_bound, aabbs);
    // aabbs.push_back(box_bound);
    return;
  }
  float *box_min = reinterpret_cast<float*>(&box_bound.minX);
  float *box_max = reinterpret_cast<float*>(&box_bound.maxX);
  uint axis = findWidest(box_bound);
  float xM = (box_min[axis] + box_max[axis]) / 2;
  uint median = split(l, r, axis, xM);

  OptixAabb left_bound = box_bound;
  OptixAabb right_bound = box_bound;
  reinterpret_cast<float*>(&left_bound.maxX)[axis] = xM;
  reinterpret_cast<float*>(&right_bound.minX)[axis] = xM;
  build(l, median, node_id*2, left_bound, aabbs);
  build(median, r, node_id*2+1, right_bound, aabbs);
}
