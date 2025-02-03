#include "flags.h"

DEFINE_uint32(buffer_size, 200000, "buffer size");
DEFINE_uint32(n_subspaces, 1, "Number of subspaces");
DEFINE_uint32(degree, 64, "degree");
DEFINE_uint32(ALGO, 1, "0 random entry, 1 rt entry");
DEFINE_uint32(search_width, 1, "search width");

/* sift1m */
// DEFINE_string(data_name, "sift1M", "name of datasets");
// DEFINE_string(data_path, "/data/myl/sift1M/sift1M_base.fvecs", "path of datasets");
// DEFINE_string(query_path, "/data/myl/sift1M/sift1M_query.fvecs", "path of queries");
// DEFINE_string(gt_path, "/data/myl/sift1M/sift1M_groundtruth.ivecs", "path of the ground truth");
// // DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/sift1M_32_16.nsw", "path of the graph");
// DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/sift1M_64_32.nsw", "path of the graph");
// // DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/sift1M_degree32.nsw", "path of the graph");
// // DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/sift1M_last64_64_16.nsw", "path of the graph");

/* sift10m */
DEFINE_string(data_name, "sift10M", "name of datasets");
DEFINE_string(data_path, "/data/myl/sift10M/sift10M_base.fbin", "path of datasets");
DEFINE_string(query_path, "/data/myl/sift10M/sift10M_query.bvecs", "path of queries");
DEFINE_string(gt_path, "/data/myl/sift10M/sift10M_groundtruth.ivecs", "path of the ground truth");
DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/sift10M_32_16.nsw", "path of the graph");
// DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/sift10M_degree32.nsw", "path of the graph");
// DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/sift10M_128_16.nsw", "path of the graph");
// DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/sift10M_128_8.nsw", "path of the graph");

/* sift100m */
// DEFINE_string(data_name, "/data/myl/sift1B/sift100M_index/pca", "name of datasets");
// DEFINE_string(data_path, "/data/myl/sift1B/bigann_base.bvecs", "path of datasets");
// DEFINE_string(query_path, "/data/myl/sift1B/bigann_query.bvecs", "path of queries");
// DEFINE_string(gt_path, "/data/myl/sift1B/gnd/idx_100M.ivecs", "path of the ground truth");
// DEFINE_string(graph_path, "/data/myl/sift1B/sift100M_index/sift100M_degree64.nsw", "path of the graph");

DEFINE_uint32(n_candidates, 64, "candidates size");
DEFINE_uint32(max_hits, 1, "max hits");
DEFINE_double(expand_ratio, 0.8, "expand ratio");
DEFINE_double(point_ratio, 0.0025, "point ratio");
DEFINE_uint32(topk, 10, "topk");
DEFINE_uint32(n_clusters, 10, "n_clusters");
// DEFINE_double(grid_size, 16.0, "grid size");