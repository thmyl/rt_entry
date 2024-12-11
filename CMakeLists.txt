cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(rt_filter C CXX CUDA)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include("cmake/FindGFlags.cmake")
if (NOT GFLAGS_FOUND)
    message(STATUS "gflags not found, build without gflags")
else ()
    include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
endif ()


find_package(OptiX REQUIRED)
if(OptiX_FOUND)
    include_directories(${OptiX_INCLUDE_DIR})
    message(STATUS "Optix Found")
    message(STATUS ${OptiX_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Optix cannot found")
endif()

# Find at least a 5.0 version of CUDA.
find_package(CUDA REQUIRED)
# 启用CUDA语言
if(CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
  message(STATUS "CUDA Found")
  message(STATUS ${CUDA_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "CUDA cannot found")
endif()

find_package(OpenMP REQUIRED)

include(${PROJECT_SOURCE_DIR}/cmake/configure_optix.cmake)
# include(${PROJECT_SOURCE_DIR}/cmake/FindOptiX.cmake)

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

# 添加可执行文件
# add_executable(test ${SOURCES})
add_executable(test
  ${embedded_ptx_code}
  Timing.cpp
  main.cpp
  aabb.cu
  entry.cu
  entry_helper.cu
  matrix.cu
  file_read.cu
  pca.cpp
  raytracing.cpp
  kdtree.cpp
  flags.cpp
  graph.cu
  graph_helper.cu
  graph_search.cuh
)

# target_link_libraries(test
#   ${optix_LIBRARY}
#   ${CUDA_LIBRARIES}
#   # ${CUDA_CUDA_LIBRARY}
#   )

target_include_directories(test PRIVATE
    ${CMAKE_SOURCE_DIR}/deps/cxxopts/include
)

target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
	-Wall -Wextra -Wpedantic
	${OpenMP_CXX_FLAGS}
	-std=c++14
>)

target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --use_fast_math
    --expt-relaxed-constexpr
    -O3
    -Xcompiler=-fopenmp
    -std=c++14
    -gencode arch=compute_75,code=sm_75
    -lineinfo
>)

target_link_libraries(test
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    OpenMP::OpenMP_CXX
    cxxopts
    cublas
    ${GFLAGS_LIBRARIES}
)

add_subdirectory(${CMAKE_SOURCE_DIR}/deps/cxxopts)
# 使用CUDA编译器对algo.cu进行编译
# set_source_files_properties(algo.cu PROPERTIES LANGUAGE CUDA)