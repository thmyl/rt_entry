#pragma once
#include <gflags/gflags.h>

#include "flags.h"

// DECLARE_string(variant);
// DECLARE_string(datafile);
// DECLARE_string(queryfile);
// DECLARE_string(gtfile);
// DECLARE_int32(subspace);
// DECLARE_double(point_ratio);
// DECLARE_double(region_expand);
// DECLARE_int32(max_regions);
// DECLARE_double(candidate_ratio);
// DECLARE_int32(grid_size);
// DECLARE_int32(search_steps);

DECLARE_uint32(buffer_size);
DECLARE_uint32(n_subspaces);
DECLARE_string(data_name);
DECLARE_string(data_path);
DECLARE_string(query_path);
DECLARE_string(gt_path);
DECLARE_uint32(entries_size);
DECLARE_uint32(max_hits);
DECLARE_double(expand_ratio);
DECLARE_double(point_ratio);
DECLARE_double(grid_size);