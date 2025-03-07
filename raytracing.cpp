#include <optix_function_table_definition.h>
#include "raytracing.h"
#include "head.h"

extern "C" char embedded_ptx_code[];

void OptiXRT::CreateContext(){
  printf("Create context\n");

  CUDA_CHECK(cudaFree(0));
  OPTIX_CHECK(optixInit());
  cuCtxGetCurrent(&cuda_context_);
  CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
  OptixDeviceContextOptions options={};
  options.logCallbackFunction=nullptr; // &context_log_cb;
  options.logCallbackLevel=4;
  // cuda_context_=0;
  OPTIX_CHECK(optixDeviceContextCreate(cuda_context_,&options,&optix_context_));
}

void OptiXRT::CreateModule(){
  printf("Create module\n");

  module_compile_options_.maxRegisterCount=OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options_.optLevel=OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_compile_options_.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  pipeline_compile_options_.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_compile_options_.usesMotionBlur=false;
  pipeline_compile_options_.numPayloadValues=2; // ray payload
  pipeline_compile_options_.numAttributeValues=0; // attribute in optixReportIntersection()
  pipeline_compile_options_.exceptionFlags=OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options_.pipelineLaunchParamsVariableName="params";

  const std::string ptx_code=embedded_ptx_code;
  char log[2048];
  size_t sizeof_log=sizeof(log);

  OPTIX_CHECK(optixModuleCreateFromPTX(
    optix_context_,
    &module_compile_options_,
    &pipeline_compile_options_,
    ptx_code.c_str(),
    ptx_code.size(),
    log,
    &sizeof_log,
    &optix_module_
  ));
}

void OptiXRT::CreateProgramGroups(){
  printf("Create program groups\n");

  char log[2048];
  size_t sizeof_log=sizeof(log);
  OptixProgramGroupOptions program_group_options={};

  OptixProgramGroupDesc raygen_prog_group_desc={};
  raygen_prog_group_desc.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_prog_group_desc.raygen.module=optix_module_;
  raygen_prog_group_desc.raygen.entryFunctionName="__raygen__rg";
  OPTIX_CHECK(optixProgramGroupCreate(
    optix_context_,&raygen_prog_group_desc,1, // num program groups
    &program_group_options,log,&sizeof_log,&raygen_prog_group_
  ));

  OptixProgramGroupDesc miss_prog_group_desc={};
  miss_prog_group_desc.kind=OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_prog_group_desc.miss.module=nullptr;
  miss_prog_group_desc.miss.entryFunctionName = nullptr;
  OPTIX_CHECK(optixProgramGroupCreate(
    optix_context_,&miss_prog_group_desc,1, // num program groups
    &program_group_options,log,&sizeof_log,&miss_prog_group_
  ));

  OptixProgramGroupDesc hit_prog_group_desc={};
  hit_prog_group_desc.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hit_prog_group_desc.hitgroup.moduleAH=optix_module_;//optix_module_;
  hit_prog_group_desc.hitgroup.entryFunctionNameAH="__anyhit__ah";
  hit_prog_group_desc.hitgroup.moduleCH=nullptr;
  hit_prog_group_desc.hitgroup.entryFunctionNameCH=nullptr;
  hit_prog_group_desc.hitgroup.moduleIS=optix_module_;
  hit_prog_group_desc.hitgroup.entryFunctionNameIS="__intersection__aabb";

  OPTIX_CHECK(optixProgramGroupCreate(
    optix_context_,&hit_prog_group_desc,1,  // num program groups
    &program_group_options,log,&sizeof_log,&hitgroup_prog_group_
  ));
}

void OptiXRT::CreatePipeline(){
  printf("Create pipeline\n");

  char log[2048];
  size_t sizeof_log=sizeof(log);

  const uint32_t max_trace_depth=2;
  pipeline_link_options_.maxTraceDepth=1; // maximum recursion depth setting for recursive ray tracing
  pipeline_link_options_.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_FULL;// pipeline level settings for debugging
  OptixProgramGroup program_groups[3]={
    raygen_prog_group_,
    miss_prog_group_,
    hitgroup_prog_group_
  };

  OPTIX_CHECK(optixPipelineCreate(
    optix_context_,
    &pipeline_compile_options_,
    &pipeline_link_options_,
    program_groups,
    sizeof(program_groups)/sizeof(program_groups[0]),
    log,
    &sizeof_log,
    &optix_pipeline_
  ));
  // ========================
  OptixStackSizes stack_sizes={};
  for(auto& prog_group : program_groups){
    OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group,&stack_sizes));
  }
  uint32_t direct_callable_stack_size_from_traversal;
  uint32_t direct_callable_stack_size_from_state;
  uint32_t continuation_stack_size;
  OPTIX_CHECK(optixUtilComputeStackSizes(
    &stack_sizes,
    max_trace_depth,
    0,0, // maxCCDepth, maxDCDepth
    &direct_callable_stack_size_from_traversal,
    &direct_callable_stack_size_from_state,
    &continuation_stack_size
  ));
  OPTIX_CHECK(optixPipelineSetStackSize(optix_pipeline_,
    direct_callable_stack_size_from_traversal,
    direct_callable_stack_size_from_state,
    continuation_stack_size,
    1  // maxTraversableDepth
  ));
}

void OptiXRT::CreateSBT(){
  printf("Create SBT\n");

  // build raygen record
  CUdeviceptr d_raygen_record=0;
  const size_t raygen_record_size=sizeof(RayGenRecord);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record),raygen_record_size));
  RayGenRecord rg_record;
  OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group_,&rg_record));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record),&rg_record,raygen_record_size,cudaMemcpyHostToDevice));
  // build miss record
  CUdeviceptr d_miss_record=0;
  const size_t miss_record_size=sizeof(MissRecord);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record),miss_record_size));
  MissRecord ms_record;
  OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group_,&ms_record));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record),&ms_record,miss_record_size,cudaMemcpyHostToDevice));
  // build hitgroup record
  CUdeviceptr d_hitgroup_record=0;
  const size_t hitgroup_record_size=sizeof(HitGroupRecord);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record),hitgroup_record_size));
  HitGroupRecord hg_record;
  OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_,&hg_record));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record),&hg_record,hitgroup_record_size,cudaMemcpyHostToDevice));
  // build sbt
  sbt_.raygenRecord=d_raygen_record;
  sbt_.missRecordBase=d_miss_record;
  sbt_.missRecordStrideInBytes=sizeof(MissRecord);
  sbt_.missRecordCount=1;
  sbt_.hitgroupRecordBase=d_hitgroup_record;
  sbt_.hitgroupRecordStrideInBytes=sizeof(HitGroupRecord);
  sbt_.hitgroupRecordCount=1;
}

void OptiXRT::Setup(){
  CreateContext();
  CreateModule();
  CreateProgramGroups();
  CreatePipeline();
  CreateSBT();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params_ptr_),sizeof(LaunchParams)));
  cudaFree(0);
}

void OptiXRT::BuildAccel(OptixAabb* d_aabbs, int num){
  //* switch optix input 
  
  //TODO: choose aabb
#ifdef _DETAIL_
  std::cout<<"primitive type = aabbs"<<std::endl;
#endif
  //* translate
  int aabbs_num=num;

  CUdeviceptr d_aabb_buffer=reinterpret_cast<CUdeviceptr>(d_aabbs);
  // build_input_buffer_.emplace_back(d_aabb_buffer);

  OptixBuildInput build_input={};
  build_input.type=OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  build_input.customPrimitiveArray.aabbBuffers=&d_aabb_buffer;
  build_input.customPrimitiveArray.numPrimitives=aabbs_num; //* number of aabb
  const uint32_t aabb_input_flags[1]={OPTIX_GEOMETRY_FLAG_NONE}; //{OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};//
  build_input.customPrimitiveArray.flags=aabb_input_flags;
  build_input.customPrimitiveArray.numSbtRecords=1;

  // Use default options for simplicity.
  OptixAccelBuildOptions accel_options={};
  accel_options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;// OPTIX_BUILD_FLAG_ALLOW_COMPACTION, OPTIX_BUILD_FLAG_NONE
  accel_options.operation=OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(
    optix_context_,
    &accel_options,
    &build_input,
    1, // Number of build inputs
    &gas_buffer_sizes
  ));

  CUdeviceptr d_temp_buffer_gas;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas),gas_buffer_sizes.tempSizeInBytes));
  CUdeviceptr d_gas_output_buffer;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer),gas_buffer_sizes.outputSizeInBytes));
  CUDA_SYNC_CHECK();
  cudaEvent_t build_start,build_end;
  CUDA_CHECK(cudaEventCreate(&build_start));
  CUDA_CHECK(cudaEventCreate(&build_end));
  CUDA_CHECK(cudaEventRecord(build_start,cuda_stream_));
  OPTIX_CHECK(optixAccelBuild(
    optix_context_, 
    cuda_stream_, // CUDA stream
    &accel_options,
    &build_input,
    1, // num build inputs
    d_temp_buffer_gas,
    gas_buffer_sizes.tempSizeInBytes,
    d_gas_output_buffer,
    gas_buffer_sizes.outputSizeInBytes,
    &gas_handle_,
    nullptr,//&emit_property, // emitted property list
    0 // num of emitted properties
  ));
  CUDA_CHECK(cudaEventRecord(build_end,cuda_stream_));
  CUDA_CHECK(cudaEventSynchronize(build_end));
  CUDA_CHECK(cudaEventElapsedTime(&build_time_,build_start,build_end));
  CUDA_CHECK(cudaEventDestroy(build_start));
  CUDA_CHECK(cudaEventDestroy(build_end));
  // cudaDeviceSynchronize();
  CUDA_SYNC_CHECK();

  /* clear up */
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
  //* free the memory of buffer
  // printf("size of build_input_buffer_ = %d\n",build_input_buffer_.size());
  // for(auto e:build_input_buffer_)
  //     CUDA_CHECK(cudaFree(reinterpret_cast<void*>(e)));

  d_gas_output_buffer_=d_gas_output_buffer;
// #ifdef _DETAIL_
  printf("BVH Building Time = %f ms\n",build_time_);
// #endif
}

void OptiXRT::CleanUp(){
    printf("Clean up RT ...\n");

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt_.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt_.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt_.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer_)));

    OPTIX_CHECK(optixPipelineDestroy(optix_pipeline_));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group_));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group_));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group_));
    OPTIX_CHECK(optixModuleDestroy(optix_module_));
    CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
    OPTIX_CHECK(optixDeviceContextDestroy(optix_context_));
// #ifdef _DETAIL_
    printf("Finish cleaning up RT\n");
// #endif
}

void OptiXRT::search(float* d_queries, int nq, int offset, int dim, int* hits){
  // Timing::startTiming("before_optixSearch1");
  h_params_.handle = gas_handle_;
  h_params_.queries = d_queries;
  h_params_.hits = hits;
  h_params_.dim = dim;
  h_params_.offset = offset;
  // Timing::stopTiming();

  // Timing::startTiming("before_optixSearch2");
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params_ptr_),&h_params_,sizeof(LaunchParams),cudaMemcpyHostToDevice));
  // Timing::stopTiming();

  // Timing::startTiming("before_optixSearch3");
  // CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params_ptr_),&h_params_,sizeof(LaunchParams),cudaMemcpyHostToDevice));
  // Timing::stopTiming();

  // Timing::startTiming("before_optixSearch4");
  // CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params_ptr_),&h_params_,sizeof(LaunchParams),cudaMemcpyHostToDevice));
  // Timing::stopTiming();

  
  // LaunchParams* h_params_pinned = nullptr; 
  // cudaHostAlloc(&h_params_pinned, sizeof(LaunchParams), cudaHostAllocDefault); // 复制数据到页锁定内存 
  // Timing::startTiming("before_optixSearch5");
  // *h_params_pinned = h_params_; // 执行数据传输 
  
  // CUDA_CHECK(cudaMemcpy(d_params_ptr_, h_params_pinned, sizeof(LaunchParams), cudaMemcpyHostToDevice)); // 释放页锁定内存 cudaFreeHost(h_params_pinned);
  // Timing::stopTiming();

  // cudaEvent_t launch_start,launch_end;
  // CUDA_CHECK(cudaEventCreate(&launch_start));
  // CUDA_CHECK(cudaEventCreate(&launch_end));
  // CUDA_CHECK(cudaEventRecord(launch_start,cuda_stream_));
  // Timing::startTiming("optixLaunch");
  OPTIX_CHECK(optixLaunch(
    optix_pipeline_,
    cuda_stream_,
    reinterpret_cast<CUdeviceptr>(d_params_ptr_),
    sizeof(LaunchParams),
    &sbt_,
    nq,
    1,1
  ));
  // Timing::stopTiming();
  // CUDA_CHECK(cudaEventRecord(launch_end,cuda_stream_));
  // CUDA_CHECK(cudaEventSynchronize(launch_end));
  // CUDA_CHECK(cudaEventElapsedTime(&search_time_,launch_start,launch_end));
  // CUDA_CHECK(cudaEventDestroy(launch_start));
  // CUDA_CHECK(cudaEventDestroy(launch_end));
  // printf("RT Search Time = %f ms\n",search_time_);
}

