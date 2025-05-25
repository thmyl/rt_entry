#pragma once
#include <gflags/gflags.h>

#include "flags.h"

DECLARE_int32(buffer_size);
DECLARE_int32(n_subspaces);
DECLARE_int32(degree);
DECLARE_int32(ALGO);
DECLARE_int32(search_width);
DECLARE_string(data_name);
DECLARE_string(data_path);
DECLARE_string(query_path);
DECLARE_string(gt_path);
DECLARE_string(graph_path);
DECLARE_int32(n_candidates);
DECLARE_int32(max_hits);
DECLARE_double(expand_ratio);
DECLARE_double(point_ratio);
DECLARE_int32(topk);
DECLARE_int32(max_iter);
// DECLARE_double(grid_size);