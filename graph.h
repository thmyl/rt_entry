#pragma once
#include "entry.h"
#include "pca.h"
#include "cache/page_cache.h"

struct DistPair{
	float dist;
	int id;
};

struct ClusterData {
    int K;
    std::vector<int> labels;
    std::vector<float> centroids;
    std::vector<std::vector<int>> cluster_points;
};

class Graph{
public:
	Graph(){}
	Graph(int n_subspaces_, int buffer_size_, int n_candidates_, int max_hits_, double expand_ratio_, double point_ratio_,
				std::string data_name_, std::string &data_path_, std::string &query_path_, std::string &gt_path_, std::string &centroids_path_, std::string &graph_path_, int ALGO_, int search_width_, int topk_, int max_iter_, int t_, int n_cluster_, int page_size_, int n_page_);
	~Graph();
	void Init_entry();
	void Search();
	void Input();
	void Projection();
	void CleanUp();
	void check_entries(thrust::device_vector<int> &d_gt_);
	void check_results(thrust::device_vector<int> &d_gt_);
	void check_results(thrust::device_vector<int> &d_gt_, int search_k);
	void RB_Graph();
	void GraphSearchBatch(int query_offset, int batch_size, cudaStream_t stream = nullptr);
	void parallel_reorder(int* candidates, int* results, int n_candidates, int topk, int dim_, int nq, float* queries, int np, float* points, DistPair* candidates_dist);
	void CopyHostToDevice(thrust::host_vector<float> &h_data, thrust::device_vector<float> &d_data, int n, int d, int d_);

public:
	RT_Entry*			rt_entry;
	std::string 		data_name;
	char*				datafile;
	char*				queryfile;
	char*				gtfile;
	char* 				graphfile;
	char* 				centroids_file;
	std::string         linear_params_path;
	std::string          rotation_matrix_path;
	std::string          mean_matrix_path;
	std::string          pca_base_path;
	cublasHandle_t       handle_;
	
	thrust::host_vector<float> h_points_;
	thrust::host_vector<float> h_queries_;
	thrust::host_vector<int> h_gt_;
	thrust::host_vector<int> h_graph_;
	
	thrust::device_vector<float> d_points_;
	thrust::device_vector<float> d_queries_;
	thrust::device_vector<int> d_gt_;
	thrust::device_vector<int> d_graph_;
	
	thrust::device_vector<float> d_rotation;
	
	thrust::device_vector<float> d_pca_queries;
	thrust::device_vector<float> d_pca_points;
	thrust::device_vector<float> d_pca_queries_full;

	thrust::device_vector<int> d_entries; //已弃用
	thrust::device_vector<float> d_entries_dist; //已弃用

	thrust::device_vector<int> d_results;
	thrust::host_vector<int> h_results;

	//reorder
	thrust::device_vector<int> d_candidates;
	thrust::host_vector<int> h_candidates;

    thrust::host_vector<DistPair> candidates_dist;

    // 双 buffer page cache，用于与 graph search 交替配合
    PageCache* page_caches[2];
	int page_size;
	int n_page;
	int n_cluster;
	int dim_partial;
	int cluster_top_t;
    int batch_size = 5;

    thrust::device_vector<float> d_centroids_matrix;
    thrust::device_vector<float> d_centroid_norms;
    thrust::device_vector<float> d_query_norms;
    thrust::device_vector<float> d_query_centroid_dists;
    std::vector<int> h_query_top_clusters;
    thrust::device_vector<int> d_query_top_clusters;
    std::vector<std::vector<int> > batch_cluster_ids;
	// std::vector<std::vector<int> > query_batch_ids;//每个batch中的query id
	std::vector<int> query_batch_ids;//每个batch中的query id
	int* d_query_batch_ids;
	std::vector<float> linear_w_host;
	std::vector<float> linear_b_host;
	thrust::device_vector<float> d_linear_w;
	thrust::device_vector<float> d_linear_b;
	int linear_params_dim = 0;

	ClusterData cluster_data;
	thrust::host_vector<int> h_query_top1_cluster;
	thrust::device_vector<int> d_query_top1_cluster;
	thrust::host_vector<int> h_cluster_entries;
	thrust::device_vector<int> d_cluster_entries;
private:
	int								 ALGO = 1;
	int                 dim_;
	int                 nq;
	int                 np;
	int                 gt_k;
	int                 n_entries = 64;
	int								 degree;
	int                 topk = 3;
	int 							   n_candidates;
	int								 search_width = 1;
	int 							   offset_shift_;
	float 							 point_ratio;
	int 								 n_hits;
	int 								 max_iter;

private:
    void compute_query_cluster_top();
    void build_query_batches();
    // 为指定 buffer 预取某个 batch 需要的 cluster 到对应的 page cache 中
	void prefetch_batch_clusters(int buffer_id, int batch_index, int query_offset, int batch_size, 
                                 cudaStream_t stream = nullptr);
	void load_linear_params();
	void build_query_batches_gpu();
};