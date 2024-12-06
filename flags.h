#pragma once
#include <gflags/gflags.h>

#include "flags.h"

DECLARE_uint32(buffer_size);
DECLARE_uint32(n_subspaces);
DECLARE_uint32(degree);
DECLARE_uint32(ALGO);
DECLARE_uint32(search_width);
DECLARE_string(data_name);
DECLARE_string(data_path);
DECLARE_string(query_path);
DECLARE_string(gt_path);
DECLARE_string(graph_path);
DECLARE_uint32(n_entries);
DECLARE_uint32(max_hits);
DECLARE_double(expand_ratio);
DECLARE_double(point_ratio);
// DECLARE_double(grid_size);