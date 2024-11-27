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

#define KDTREE
#define DIM 128
#define COUNT_DIM 32
// #define GRID