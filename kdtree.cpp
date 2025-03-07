#include "kdtree.h"

Kdtree::Kdtree(float3* _data, int _n, int _max_node, OptixAabb world_bound, thrust::host_vector<OptixAabb> &aabbs){
  max_node = _max_node;
  printf("max_node = %d\n", max_node);
  n = _n;
  belong = new int[n];
  data.resize(n);
  for(int i=0; i<n; i++){
    data[i].resize(3);
    data[i][0] = _data[i].x;
    data[i][1] = _data[i].y;
    data[i][2] = _data[i].z;
  }
  id = new int[n];
  for(int i=0; i<n; i++)id[i] = i;
  // root = new Kdnode;
  // build(0, n, 0, world_bound, aabbs);
  buildWithStack(0, n, 0, world_bound, aabbs);
}

Kdtree::~Kdtree(){
  // for(int i=0; i<n; i++)delete[] data[i];
  // delete[] data;
  delete[] id;
  delete[] belong;
}

int findWidest(OptixAabb box_bound){
  float max_width = 0;
  int axis = 0;
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

int Kdtree::split(int l, int r, int axis, float xM){
  r = r-1;
  int st_l = l, st_r = r;
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

void Kdtree::add_aabb(int l, int r, OptixAabb box_bound, thrust::host_vector<OptixAabb> &aabbs){
  int aabb_id = aabbs.size();
  for(int i=l; i<r; i++){
    belong[id[i]] = aabb_id;
  }
  aabbs.push_back(box_bound);
}

/*void Kdtree::computeAabbPid(thrust::device_vector<int> &aabb_pid, thrust::device_vector<int> &prefix_sum, int n_aabbs){
  thrust::host_vector<int> h_aabb_pid(n);
  thrust::host_vector<int> h_prefix_sum(n_aabbs + 1);
  int* count = new int[n_aabbs];
  for(int i=0; i<n_aabbs; i++)count[i] = 0;
  for(int i=0; i<n; i++)count[belong[i]]++;
  h_prefix_sum[0] = 0;
  for(int i=0; i<n_aabbs; i++){
    h_prefix_sum[i+1] = h_prefix_sum[i] + count[i];
  }
  delete[] count;
  thrust::copy(h_prefix_sum.begin(), h_prefix_sum.end(), prefix_sum.begin());
  for(int i=0; i<n; i++){
    h_aabb_pid[h_prefix_sum[belong[i]]] = i;
    h_prefix_sum[belong[i]]++;
  }
  thrust::copy(h_aabb_pid.begin(), h_aabb_pid.end(), aabb_pid.begin());
}*/

void Kdtree::computeAabbPid(thrust::host_vector<int> &h_aabb_pid, int n_aabbs){
  // thrust::host_vector<int> h_aabb_pid(n_aabbs * max_node, n);
  h_aabb_pid.resize(n_aabbs * max_node);
  thrust::fill(h_aabb_pid.begin(), h_aabb_pid.end(), n);
  std::vector<int> count(n_aabbs, 0);
  for(int i=0; i<n; i++){
    int aabb_id = belong[i];
    h_aabb_pid[aabb_id * max_node + count[aabb_id]] = i;
    count[aabb_id]++;
  }
}

float Kdtree::findxM(float l, float r, int data_l, int data_r, int axis){
  float xM = (l+r)/2.0;
  float mid;
  int _n = data_r - data_l;
  int cnt = 0;
  float _eps = std::max((r-l)/1000, 0.01f);
  // if(_n > 10*max_node) return (l+r)/2.0;
  while(r-l > _eps){
    mid = (r+l)/2.0;

    cnt = 0;
    for(int i=data_l; i<data_r; i++){
      if(data[i][axis] <= mid) cnt++;
    }
    if(cnt <= _n/2) {
      l=mid;
      xM=mid;
      if(cnt == _n/2)break;
    }
    else r=mid;
  }
  return xM;
}

void Kdtree::tight_box(int l, int r, float *box_min, float *box_max){
  for(int i=0; i<3; i++){
    box_min[i] = data[l][i];
    box_max[i] = data[l][i];
  }
  for(int i=l+1; i<r; i++){
    for(int j=0; j<3; j++){
      box_min[j] = std::min(box_min[j], data[i][j]);
      box_max[j] = std::max(box_max[j], data[i][j]);
    }
  }
}

void Kdtree::buildWithStack(int l, int r, int node_id, OptixAabb box_bound, thrust::host_vector<OptixAabb> &aabbs) {
  // Define a structure to hold the information needed for each build step
  printf("building kd-tree with stack\n");
  struct BuildTask {
      int l;
      int r;
      int node_id;
      OptixAabb box_bound;
  };

  // Use a stack to simulate recursion
  std::stack<BuildTask> taskStack;

  // Push the initial task onto the stack
  taskStack.push({l, r, node_id, box_bound});

  while (!taskStack.empty()) {
      // Pop the top task from the stack
      BuildTask task = taskStack.top();
      taskStack.pop();

      int current_l = task.l;
      int current_r = task.r;
      int current_node_id = task.node_id;
      // printf("%d %d %d\n", current_l, current_r, current_node_id);
      OptixAabb current_box_bound = task.box_bound;

      if (current_l >= current_r) continue;  // Skip invalid ranges

      float *box_min = reinterpret_cast<float*>(&current_box_bound.minX);
      float *box_max = reinterpret_cast<float*>(&current_box_bound.maxX);
      tight_box(current_l, current_r, box_min, box_max);

      if (current_r - current_l <= max_node) {
          add_aabb(current_l, current_r, current_box_bound, aabbs);
          continue;
      }

      int axis = findWidest(current_box_bound);
      // float xM = (box_min[axis] + box_max[axis]) / 2;
      float xM = findxM(box_min[axis], box_max[axis], current_l, current_r, axis);
      int median = split(current_l, current_r, axis, xM);

      OptixAabb left_bound = current_box_bound;
      OptixAabb right_bound = current_box_bound;
      reinterpret_cast<float*>(&left_bound.maxX)[axis] = xM + eps;
      reinterpret_cast<float*>(&right_bound.minX)[axis] = xM - eps;

      // Push the right subtree task onto the stack first so that it is processed after the left subtree
      taskStack.push({median, current_r, current_node_id * 2 + 1, right_bound});
      taskStack.push({current_l, median, current_node_id * 2, left_bound});
  }
}