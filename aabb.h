#pragma once
#include <optix.h>
#include "head.h"

void TraversePointsInRegion(thrust::device_vector<float3> &d_points, thrust::device_vector<OptixAabb> &d_aabbs, 
                            thrust::device_vector<uint> &aabb_pid, thrust::device_vector<uint> &prefix_sum);

void computeMinMax(OptixAabb &world_bounds, thrust::host_vector<float3> h_points);

void generate_uniform_aabbs(thrust::host_vector<OptixAabb> &h_aabbs, uint grid_size, OptixAabb world_bounds, uint n_aabbs);
void expandAabb(thrust::host_vector<OptixAabb> &h_aabbs, float expand_ratio);
void expandAabb_uniform(thrust::host_vector<OptixAabb> &h_aabbs, float3 expand);