#pragma once
#include "entry.h"
#include "pca.h"

class Graph{
public:
	Graph(){}
	Graph(uint n_subspaces_, uint buffer_size_, uint n_candidates_, uint max_hits_, double expand_ratio_, double point_ratio_,
				std::string data_name_, std::string &data_path_, std::string &query_path_, std::string &gt_path_, std::string &graph_path_, uint ALGO_, uint search_width_, uint topk_);
	~Graph();
	void Init_entry();
	void Search();
	void Input();
	void Projection();
	void CleanUp();
	void check_entries(thrust::device_vector<uint> &d_gt_);
	void check_results(thrust::device_vector<uint> &d_gt_);
	void RB_Graph();
	void GraphSearch();

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
	thrust::host_vector<uint> h_gt_;
	thrust::host_vector<uint> h_graph_;
	
	thrust::device_vector<float> d_points_;
	thrust::device_vector<float> d_queries_;
	thrust::device_vector<uint> d_gt_;
	thrust::device_vector<uint> d_graph_;
	
	thrust::device_vector<float> d_rotation;
	
	thrust::device_vector<float> d_pca_queries;
	thrust::device_vector<float> d_pca_points;

	thrust::device_vector<uint> d_entries;
	thrust::device_vector<float> d_entries_dist;

	thrust::device_vector<uint> d_results;

	//reorder
	thrust::device_vector<uint> candidates_;
	thrust::device_vector<uint> map_id;
	thrust::device_vector<float> full_vector;
	thrust::device_vector<uint> unique_candidates;

private:
	uint								 ALGO = 1;
	uint                 dim_;
	uint                 nq;
	uint                 np;
	uint                 gt_k;
	uint                 n_entries = 64;
	uint								 degree;
	uint                 topk = 3;
	uint 							   n_candidates;
	uint								 search_width = 1;
	uint 							   offset_shift_;
	float 							 point_ratio;
	uint 								 n_hits;
};