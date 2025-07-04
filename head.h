#include <optix_types.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#include <thrust/sort.h>
#include <thrust/set_operations.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <gflags/gflags.h>


#include "helper.h"
#include "Timing.h"
#include "kdtree.h"
#include "flags.h"

// #define KDTREE
#define DIM 64
#define ENTRY_DIM 32 //dimensions used in entry search
#define GRAPH_DIM 64 //dimensions used in graph search
#define USE_L2_DIST_
// #define REORDER
#define DETAIL //Whether to output detailed information
// #define GRID