#pragma once
#include "entry.h"
#include "pca.h"

class Graph{
public:
	Graph(){}
	Graph(uint n_subspaces_, uint buffer_size_, uint entries_size_, uint max_hits_, double expand_ratio_, double point_ratio_,
				std::string data_name_, std::string &data_path_, std::string &query_path_, std::string &gt_path_);
	~Graph();
	void Init_entry();
	void Search();
	void Input();
	void Projection();
	void CleanUp();
	void check_entries(thrust::device_vector<uint> &d_gt_);

public:
	RT_Entry* 					 rt_entry;
	std::string 				 data_name;
	char* 							 datafile;
	char* 							 queryfile;
	char* 							 gtfile;
	std::string          rotation_matrix_path;
	std::string          mean_matrix_path;
	std::string          pca_base_path;
	cublasHandle_t       handle_;
	
	thrust::host_vector<float> h_points_;
	thrust::host_vector<float> h_queries_;
	thrust::host_vector<uint> h_gt_;
	
	thrust::device_vector<float> d_queries_;
	thrust::device_vector<uint> d_gt_;
	
	thrust::device_vector<float> d_rotation;
	
	thrust::device_vector<float> d_pca_queries;
	thrust::device_vector<float> d_pca_points;

	thrust::device_vector<uint> d_entries;
	thrust::device_vector<float> d_entries_dist;
	uint                 dim_;
	uint                 nq;
	uint                 np;
	uint                 gt_k;
	uint                 n_entries = 64;
};