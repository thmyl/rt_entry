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

void generate_uniform_aabbs(thrust::host_vector<OptixAabb> &h_aabbs, uint grid_size, OptixAabb world_bounds, uint n_aabbs){
  h_aabbs.resize(n_aabbs);
  float3 delta = make_float3((world_bounds.maxX - world_bounds.minX) / grid_size, (world_bounds.maxY - world_bounds.minY) / grid_size, (world_bounds.maxZ - world_bounds.minZ) / grid_size);
  printf("delta: %f %f %f\n", delta.x, delta.y, delta.z);
  for(int i=0; i<grid_size; i++){
    for(int j=0; j<grid_size; j++){
      for(int k=0; k<grid_size; k++){
        int id = i + j*grid_size + k*grid_size*grid_size;
        h_aabbs[id].minX = world_bounds.minX + i*delta.x;
        h_aabbs[id].minY = world_bounds.minY + j*delta.y;
        h_aabbs[id].minZ = world_bounds.minZ + k*delta.z;
        h_aabbs[id].maxX = world_bounds.minX + (i+1)*delta.x;
        h_aabbs[id].maxY = world_bounds.minY + (j+1)*delta.y;
        h_aabbs[id].maxZ = world_bounds.minZ + (k+1)*delta.z;
        // if(i == grid_size-1) h_aabbs[id].maxX = world_bounds.maxX + 1e-2;
        // if(j == grid_size-1) h_aabbs[id].maxY = world_bounds.maxY + 1e-2;
        // if(k == grid_size-1) h_aabbs[id].maxZ = world_bounds.maxZ + 1e-2;
      }
    }
  }
}

__global__ void CountKernel(uint np, float3 *d_points, uint n_aabbs, OptixAabb *d_aabbs, uint *n_points_per_aabb){
    uint i_aabb = blockIdx.x * blockDim.x + threadIdx.x;
    if(i_aabb < n_aabbs){
        for(int i=0; i<np; i++){
            float3 p = d_points[i];
            if(p.x >= d_aabbs[i_aabb].minX && p.x < d_aabbs[i_aabb].maxX 
            && p.y >= d_aabbs[i_aabb].minY && p.y < d_aabbs[i_aabb].maxY 
            && p.z >= d_aabbs[i_aabb].minZ && p.z < d_aabbs[i_aabb].maxZ){
                atomicAdd(&n_points_per_aabb[i_aabb], 1);
            }
        }
    }
}

__global__ void AabbPidKernel(uint np, float3* d_points, uint n_aabbs, OptixAabb *d_aabbs, uint *aabb_pid, uint *prefix_sum){
    uint i_aabb = blockIdx.x * blockDim.x + threadIdx.x;
    if(i_aabb < n_aabbs){
        uint start = prefix_sum[i_aabb];
        uint end = prefix_sum[i_aabb + 1];
        for(int i=0; i<np; i++){
            float3 p = d_points[i];
            if(p.x >= d_aabbs[i_aabb].minX && p.x < d_aabbs[i_aabb].maxX 
            && p.y >= d_aabbs[i_aabb].minY && p.y < d_aabbs[i_aabb].maxY 
            && p.z >= d_aabbs[i_aabb].minZ && p.z < d_aabbs[i_aabb].maxZ){
              if(i==791282){
                printf("point791282 in aabb%d\n", i_aabb);
                printf("aabb%d: (%f, %f, %f) - (%f, %f, %f)\n", i_aabb, d_aabbs[i_aabb].minX, d_aabbs[i_aabb].minY, d_aabbs[i_aabb].minZ, d_aabbs[i_aabb].maxX, d_aabbs[i_aabb].maxY, d_aabbs[i_aabb].maxZ);
                printf("aabb_pid[%d]: %d\n", start, i);
              }
                aabb_pid[start] = i;
                start++;
            }
        }
        if(start != end){
            printf("ERROR: start != end\n");
        }
    }
}

void TraversePointsInRegion(thrust::device_vector<float3> &d_points, thrust::device_vector<OptixAabb> &d_aabbs, thrust::device_vector<uint> &aabb_pid, thrust::device_vector<uint> &prefix_sum){
    uint n_aabbs = d_aabbs.size();
    uint np = d_points.size();
    thrust::device_vector<uint> n_points_per_aabb(n_aabbs, 0);
    uint *n_points_per_aabb_ptr = thrust::raw_pointer_cast(n_points_per_aabb.data());
    uint *aabb_pid_ptr = thrust::raw_pointer_cast(aabb_pid.data());
    float3 *points_ptr = thrust::raw_pointer_cast(d_points.data());
    OptixAabb *aabbs_ptr = thrust::raw_pointer_cast(d_aabbs.data());
    CountKernel<<<(n_aabbs + 255) / 256, 256>>>(np, points_ptr, n_aabbs, aabbs_ptr, n_points_per_aabb_ptr);
    CUDA_SYNC_CHECK();
    prefix_sum[0] = 0;
    thrust::inclusive_scan(thrust::device, n_points_per_aabb.begin(), 
                           n_points_per_aabb.end(), 
                           prefix_sum.begin() + 1);
    AabbPidKernel<<<(n_aabbs + 255) / 256, 256>>>(np, points_ptr, n_aabbs, aabbs_ptr, aabb_pid_ptr, thrust::raw_pointer_cast(prefix_sum.data()));
    CUDA_SYNC_CHECK();
    thrust::host_vector<uint> h_aabb_pid = aabb_pid;
    thrust::host_vector<uint> h_prefix_sum = prefix_sum;
    printf("prefix_sum: %d\n", h_prefix_sum[n_aabbs]);
}

void expandAabb_uniform(thrust::host_vector<OptixAabb> &h_aabbs, float3 expand){
  printf("expand: %f %f %f\n", expand.x, expand.y, expand.z);
    uint n_aabbs = h_aabbs.size();
    for(int i=0; i<n_aabbs; i++){
        h_aabbs[i].minX -= expand.x;
        h_aabbs[i].minY -= expand.y;
        h_aabbs[i].minZ -= expand.z;
        h_aabbs[i].maxX += expand.x;
        h_aabbs[i].maxY += expand.y;
        h_aabbs[i].maxZ += expand.z;
    }
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