#include "aabb.h"

void computeMinMax(OptixAabb &world_bounds, thrust::host_vector<float3> h_points){
  uint N = h_points.size();
  world_bounds.minX = h_points[0].x;
  world_bounds.minY = h_points[0].y;
  world_bounds.minZ = h_points[0].z;
  world_bounds.maxX = h_points[0].x;
  world_bounds.maxY = h_points[0].y;
  world_bounds.maxZ = h_points[0].z;
  for(int i=0; i<N; i++){
    world_bounds.minX = fminf(world_bounds.minX, h_points[i].x);
    world_bounds.minY = fminf(world_bounds.minY, h_points[i].y);
    world_bounds.minZ = fminf(world_bounds.minZ, h_points[i].z);
    world_bounds.maxX = fmaxf(world_bounds.maxX, h_points[i].x);
    world_bounds.maxY = fmaxf(world_bounds.maxY, h_points[i].y);
    world_bounds.maxZ = fmaxf(world_bounds.maxZ, h_points[i].z);
  }
  world_bounds.minX -= 1e-2;
  world_bounds.minY -= 1e-2;
  world_bounds.minZ -= 1e-2;
  world_bounds.maxX += 1e-2;
  world_bounds.maxY += 1e-2;
  world_bounds.maxZ += 1e-2;
}

void expandAabb(thrust::host_vector<OptixAabb> &h_aabbs, float expand_ratio){
  uint n_aabbs = h_aabbs.size();
  for(int i=0; i<n_aabbs; i++){
    float *box_min = reinterpret_cast<float*>(&h_aabbs[i].minX);
    float *box_max = reinterpret_cast<float*>(&h_aabbs[i].maxX);
    for(int j=0; j<3; j++){
      float width = box_max[j] - box_min[j];
      box_min[j] -= width * expand_ratio;
      box_max[j] += width * expand_ratio;
    }
  }
}