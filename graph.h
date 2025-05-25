#pragma once
#include "entry.h"
#include "pca.h"

struct Pair{
	float dist;
	int id;
};

class Graph{
public:
	Graph(){}
	Graph(int n_subspaces_, int buffer_size_, int n_candidates_, int max_hits_, double expand_ratio_, double point_ratio_,
				std::string data_name_, std::string &data_path_, std::string &query_path_, std::string &gt_path_, std::string &graph_path_, int ALGO_, int search_width_, int topk_, int max_iter_);
	~Graph();
	void Init_entry();
	void Search();
	void Input();
	void Projection();
	void CleanUp();
	void check_entries(thrust::device_vector<int> &d_gt_);
	void check_results(thrust::device_vector<int> &d_gt_);
	void RB_Graph();
	void GraphSearch();
	void parallel_reorder(int* candidates, int* results, int n_candidates, int topk, int dim_, int nq, float* queries, int np, float* points, Pair* candidates_dist);
	void CopyHostToDevice(thrust::host_vector<float> &h_data, thrust::device_vector<float> &d_data, int n, int d, int d_);

public:
	RT_Entry* 					 rt_entry;
	std::string 				 data_name;
	char* 							 datafile;
	char* 							 queryfile;
	char* 							 gtfile;
	char*  							 graphfile;
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

	thrust::device_vector<int> d_entries;
	thrust::device_vector<float> d_entries_dist;

	thrust::device_vector<int> d_results;
	thrust::host_vector<int> h_results;

	//reorder
	thrust::device_vector<int> d_candidates;
	thrust::host_vector<int> h_candidates;

	thrust::host_vector<Pair> candidates_dist;

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
};