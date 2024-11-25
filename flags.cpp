#include "flags.h"

DEFINE_uint32(buffer_size, 200000, "buffer size");
DEFINE_uint32(n_subspaces, 1, "Number of subspaces");
DEFINE_string(data_name, "sift1M", "name of datasets");
DEFINE_string(data_path, "/data/myl/sift1M/sift1M_base.fvecs", "path of datasets");
DEFINE_string(query_path, "/data/myl/sift1M/sift1M_query.fvecs", "path of queries");
DEFINE_string(gt_path, "/data/myl/sift1M/sift1M_groundtruth.ivecs", "path of the ground truth");
DEFINE_uint32(entries_size, 64, "entries size");
DEFINE_uint32(max_hits, 80, "max hits");
DEFINE_double(expand_ratio, 0.8, "expand ratio");
DEFINE_double(point_ratio, 0.0025, "point ratio");
DEFINE_double(grid_size, 16.0, "grid size");