// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include <vector_types.h>
// #include "sutil/vec_math.h"
#include <cuda_runtime.h>

#include "raytracing.h"


extern "C" {
    __constant__ LaunchParams params;
}

extern "C" __global__ void __raygen__rg(){
  const uint3 idx = optixGetLaunchIndex();
  unsigned int rayIdx = idx.x;
  // float3 ray_origin = params.queries[rayIdx];
  uint dim = params.dim;
  uint offset = rayIdx * dim;
  float3 ray_origin = make_float3(params.queries[offset], params.queries[offset + 1], params.queries[offset + 2]);
  // float3 ray_origin = make_float3(1, 1, 1);
  float3 ray_direction = make_float3(1,0,0);

  unsigned int queryIdx = rayIdx;
  float tmin = 0.f;
  float tmax = 1.e-16f;
  unsigned int id = 0;

  optixTrace(
    params.handle,
    ray_origin,
    ray_direction,
    tmin,
    tmax,
    0.0f,
    OptixVisibilityMask(255),
    OPTIX_RAY_FLAG_NONE,
    0,0,0,
    reinterpret_cast<unsigned int&>(queryIdx),
    reinterpret_cast<unsigned int&>(id)
  );
}

extern "C" __global__ void __miss__ms(){
}

extern "C" __global__ void __closesthit__ch(){

}

extern "C" __global__ void __intersection__aabb(){
  unsigned int id = optixGetPayload_1();//第i个相交的aabb
  unsigned int primIdx = optixGetPrimitiveIndex();
  unsigned int queryIdx = optixGetPayload_0();
  params.hits[queryIdx] = primIdx;
  params.entries[queryIdx] = params.aabb_entry[primIdx];
  // if(id+1 >= params.max_hits){
    optixReportIntersection( 0, 0 );
  // }
  // else optixSetPayload_1(id+1);
}

// extern "C" __global__ void __intersection__aabb(){
//   unsigned int id = optixGetPayload_1();//第i个相交的aabb
//   if(id < params.k_aabb_s){
//     unsigned int queryIdx = optixGetPayload_0();
//     unsigned int primIdx = optixGetPrimitiveIndex();
//     if(primIdx & 1) return;
//     params.results_aabb_s[queryIdx * params.k_aabb_s + id] = primIdx;
    
//     params.results_k_s[queryIdx] = id+1;
    
//     if(id+1 == params.k_aabb_s){
//       optixReportIntersection( 0, 0 );
//     }
//     else optixSetPayload_1(id+1);
//   }
// }

extern "C" __global__ void __anyhit__ah(){
  optixTerminateRay();
}