#pragma once
#include <optix.h>
#include "head.h"
void computeMinMax(OptixAabb &world_bounds, thrust::host_vector<float3> h_points);
void expandAabb(thrust::host_vector<OptixAabb> &h_aabbs, float expand_ratio);