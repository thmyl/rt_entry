#pragma once

#include "head.h"
#include "helper.h"

// ------------------------------------------------------------------
// Launch Parameters
// ------------------------------------------------------------------
struct LaunchParams{
  OptixTraversableHandle handle;
  // uint max_hits;
  uint dim;
  uint offset;
  uint* hits;
  // uint* n_hits_per_query;
  float* queries;
  uint* belong;
  uint rt_id;
};

// ------------------------------------------------------------------
// Record
// ------------------------------------------------------------------
template<typename T>
struct Record{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct EmptyData{};
typedef Record<EmptyData> EmptyRecord;

typedef EmptyRecord RayGenRecord;
typedef EmptyRecord MissRecord;
typedef EmptyRecord HitGroupRecord;

// ------------------------------------------------------------------
// OptiXRT
// ------------------------------------------------------------------
class OptiXRT{
public:
// private:
  CUcontext cuda_context_=nullptr;
  OptixDeviceContext optix_context_=0;
  OptixModule optix_module_=nullptr;

  // float aabb_side_=0.2f;
  // OptixModule aabb_module_=nullptr;
  // float3 *d_centers_=nullptr;
  // OptixAabb *d_aabbs=nullptr;

  OptixModuleCompileOptions module_compile_options_ = {};
  OptixPipelineCompileOptions pipeline_compile_options_ = {};
  OptixPipelineLinkOptions pipeline_link_options_ = {};
  OptixProgramGroup raygen_prog_group_=nullptr;
  OptixProgramGroup miss_prog_group_=nullptr;
  OptixProgramGroup hitgroup_prog_group_=nullptr;
  // optix build input buffer
  // std::vector<CUdeviceptr> build_input_buffer_;
  // scene offset
  SceneParameter scene_params_;
    
// public:
  CUstream cuda_stream_;
  OptixPipeline optix_pipeline_;
  OptixShaderBindingTable sbt_={};
  OptixTraversableHandle gas_handle_;
  CUdeviceptr d_gas_output_buffer_{};
  LaunchParams h_params_;
  LaunchParams *d_params_ptr_=nullptr;
  int use_device_id_{0};
  float search_time_{0.0f};
  float build_time_{0.0f};

// private:
  void CreateContext();
  void CreateModule();
  void CreateProgramGroups();
  void CreatePipeline();
  void CreateSBT();
// public:
  OptiXRT(const SceneParameter &parameter):scene_params_(parameter){};
  OptiXRT(){};
  ~OptiXRT(){};
  void SetDevice(int device_id);
  void Setup();
  void BuildAccel(OptixAabb*, uint);
  void search(float* d_queries, uint nq, uint offset, uint dim, uint* hits);
  void search(float* d_queries, uint nq, uint offset, uint dim, uint* hits, uint* d_belong, uint rt_id);
  void CleanUp();
  // void PrintInfo();
};