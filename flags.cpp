#include "flags.h"

DEFINE_int32(buffer_size, 200000, "buffer size");
DEFINE_int32(n_subspaces, 1, "Number of subspaces");
DEFINE_int32(degree, 64, "degree");
DEFINE_int32(ALGO, 1, "0 random entry, 1 rt entry");
DEFINE_int32(search_width, 1, "search width");

/* sift1m */
DEFINE_string(data_name, "sift1M", "name of datasets");
DEFINE_string(data_path, "/data/myl/sift1M/sift1M_base.fvecs", "path of datasets");
DEFINE_string(query_path, "/data/myl/sift1M/sift1M_query.fvecs", "path of queries");
DEFINE_string(gt_path, "/data/myl/sift1M/sift1M_groundtruth.ivecs", "path of the ground truth");
DEFINE_string(graph_path, "/data/myl/sift1M/index/sift1M_64_32_cagra.nsw", "path of the graph");
// DEFINE_string(graph_path, "/data/myl/sift1M/sift_128_64_1M.nsw", "path of the graph");
// DEFINE_string(graph_path, "/home/myl/cagra/python/sift1m_128_64.nsw", "path of the graph");
// DEFINE_string(graph_path, "/home/myl/graph/index_diskann/sift1M_64_diskann.nsw", "path of the graph");
// DEFINE_string(graph_path, "/home/myl/pcsearch/input_file/sift1M_32_16.nsw", "path of the graph");
// DEFINE_string(graph_path, "/home/myl/pcsearch/input_file/sift1M_64_32.nsw", "path of the graph");
// DEFINE_string(graph_path, "/home/myl/graph/index/sift_128_64_1M.nsw", "path of the graph");

/* sift10m */
// DEFINE_string(data_name, "sift10M", "name of datasets");
// DEFINE_string(data_path, "/data/myl/sift10M/sift10M_base.fbin", "path of datasets");
// DEFINE_string(query_path, "/data/myl/sift10M/sift10M_query.bvecs", "path of queries");
// DEFINE_string(gt_path, "/data/myl/sift10M/sift10M_groundtruth.ivecs", "path of the ground truth");
// // DEFINE_string(graph_path, "/data/myl/sift10M/sift_128_64_10M.nsw", "path of the graph");
// DEFINE_string(graph_path, "/data/myl/sift10M/index/sift10M_64_32_cagra.nsw", "path of the graph");
// // DEFINE_string(graph_path, "/home/myl/pcsearch/input_file/sift10M_32_16.nsw", "path of the graph");

/* sift100m */
// DEFINE_string(data_name, "/data/myl/sift1B/sift100M_index/pca", "name of datasets");
// // DEFINE_string(data_path, "/data/myl/sift1B/bigann_base.bvecs", "path of datasets");
// DEFINE_string(data_path, "/data/myl/sift100M/sift100M_base.fbin", "path of datasets");
// DEFINE_string(query_path, "/data/myl/sift1B/bigann_query.bvecs", "path of queries");
// DEFINE_string(gt_path, "/data/myl/sift1B/gnd/idx_100M.ivecs", "path of the ground truth");
// DEFINE_string(graph_path, "/data/myl/sift1B/sift100M_index/sift100M_degree64.nsw", "path of the graph");
// // DEFINE_string(graph_path, "/data/myl/sift100M/index/sift100M_128_64_diskann.nsw", "path of the graph");

/* deep1M */
// DEFINE_string(data_name, "deep1M", "name of datasets");
// DEFINE_string(data_path, "/data/myl/deep1M/deep1M_base.fvecs", "path of datasets");
// DEFINE_string(query_path, "/data/myl/deep1M/deep1M_queries.fvecs", "path of queries");
// DEFINE_string(gt_path, "/data/myl/deep1M/deep1M_gt.ivecs", "path of the ground truth");
// // DEFINE_string(graph_path, "/data/myl/deep1M/deep_128_64_1M.nsw", "path of the graph");
// // DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/deep1M_32_16.nsw", "path of the graph");
// // DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/deep1M_64_32.nsw", "path of the graph");
// DEFINE_string(graph_path, "/data/myl/deep1M/index/deep1M_64_32_cagra.nsw", "path of the graph");

/* deep10M */
// DEFINE_string(data_name, "deep10M", "name of datasets");
// DEFINE_string(data_path, "/data/myl/deep10M/fbin/deep10M_base.fbin", "path of datasets");
// DEFINE_string(query_path, "/data/myl/deep10M/deep10M_query.fvecs", "path of queries");
// DEFINE_string(gt_path, "/data/myl/deep10M/deep10M_groundtruth.ivecs", "path of the ground truth");
// // DEFINE_string(graph_path, "/data/myl/deep10M/index/deep10M_128_64_diskann.nsw", "path of the graph");
// DEFINE_string(graph_path, "/data/myl/deep10M/index/deep10M_128_64_cagra.nsw", "path of the graph");

/* deep100M */
// DEFINE_string(data_name, "/data/myl/deep100M/index/pca", "name of datasets");
// DEFINE_string(data_path, "/data/myl/deep100M/fbin/deep100M_base.fbin", "path of datasets");
// DEFINE_string(query_path, "/data/myl/deep100M/deep100M_query.fvecs", "path of queries");
// DEFINE_string(gt_path, "/data/myl/deep100M/deep100M_groundtruth.ivecs", "path of the ground truth");
// DEFINE_string(graph_path, "/data/myl/deep100M/index/deep100M_128_64_diskann.nsw", "path of the graph");
// // DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/deep1M_32_16.nsw", "path of the graph");
// // DEFINE_string(graph_path, "/home/myl/rt_entry/input_file/deep1M_64_32.nsw", "path of the graph");

/* gist */
// DEFINE_string(data_name, "gist", "name of datasets");
// DEFINE_string(data_path, "/data/myl/gist/gist_base.fvecs", "path of datasets");
// DEFINE_string(query_path, "/data/myl/gist/gist_query.fvecs", "path of queries");
// DEFINE_string(gt_path, "/data/myl/gist/gist_groundtruth.ivecs", "path of the ground truth");
// DEFINE_string(graph_path, "/data/myl/gist/index/gist_128_64_cagra.nsw", "path of the graph");
// // DEFINE_string(graph_path, "/home/myl/cagra/python/gist_128_64.nsw", "path of the graph");

/* crawl */
// DEFINE_string(data_name, "crawl", "name of datasets");
// DEFINE_string(data_path, "/data/myl/crawl/crawl_base.fvecs", "path of datasets");
// DEFINE_string(query_path, "/data/myl/crawl/crawl_query.fvecs", "path of queries");
// DEFINE_string(gt_path, "/data/myl/crawl/crawl_groundtruth.ivecs", "path of the ground truth");
// DEFINE_string(graph_path, "/data/myl/crawl/index/crawl_128_64_cagra.nsw", "path of the graph");

/* uqv */
// DEFINE_string(data_name, "uqv", "name of datasets");
// DEFINE_string(data_path, "/data/myl/uqv/uqv_base.fvecs", "path of datasets");
// DEFINE_string(query_path, "/data/myl/uqv/uqv_query.fvecs", "path of queries");
// DEFINE_string(gt_path, "/data/myl/uqv/uqv_groundtruth.ivecs", "path of the ground truth");
// DEFINE_string(graph_path, "/data/myl/uqv/index/uqv_128_64_cagra.nsw", "path of the graph");

/* COCO-I2I */
// DEFINE_string(data_name, "COCO-I2I", "name of datasets");
// DEFINE_string(data_path, "/data/myl/COCO-I2I/COCO-I2I_base.fvecs", "path of datasets");
// DEFINE_string(query_path, "/data/myl/COCO-I2I/COCO-I2I_query.fvecs", "path of queries");
// DEFINE_string(gt_path, "/data/myl/COCO-I2I/COCO-I2I_groundtruth.ivecs", "path of the ground truth");
// DEFINE_string(graph_path, "/data/myl/COCO-I2I/index/COCO-I2I_32_16_cagra.nsw", "path of the graph");

DEFINE_int32(n_candidates, 128, "candidates size");
DEFINE_int32(max_hits, 1, "max hits");
DEFINE_double(expand_ratio, 0.2, "expand ratio");
DEFINE_double(point_ratio, 0.000128, "point ratio");
DEFINE_int32(topk, 0, "topk");
DEFINE_int32(max_iter, 100, "max iter");
// DEFINE_double(grid_size, 16.0, "grid size");