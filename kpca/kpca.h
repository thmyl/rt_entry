#pragma once

#include <pca.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../matrix.h"

class KPCA{
public:
  KPCA();
  KPCA(float* data_, uint _nb, uint _nq, uint _dim, uint _n_clusters, uint _n_components, uint _n_iteration = 20);

  void InitLists(uint method);
  void Learn();
  void SetPcas();
  void Output();
  
  void pca_tree();
  void pca_split(PCA& pca1, PCA& pca2, std::vector<uint>& list1, std::vector<uint>& list2);
  void output_lists(const char* filename);

  void SaveLearnedData();
  void WriteCluster(const char* filename);
  void ReadCluster(const char* filename);

  uint n_clusters;
  uint n_components;
  uint nb, nq, dim;
  uint n_iteration;
  float* data;
  std::vector<PCA> pcas;
  std::vector<std::vector<uint> > lists;
  Eigen::MatrixXd B;
  Eigen::MatrixXd B_pca;
  Eigen::MatrixXd B_reconstructed;

//save the results
  std::vector<thrust::host_vector<uint> > h_cluster_lists;
	std::vector<thrust::device_vector<uint> > d_cluster_lists;

	std::vector<thrust::host_vector<float> > h_rotation;
	std::vector<thrust::device_vector<float> > d_rotation;

	std::vector<thrust::host_vector<float> > h_means;
  std::vector<thrust::host_vector<float> > h_transforms;
  std::vector<thrust::host_vector<float> > h_reconstructed;
	std::vector<thrust::device_vector<float> > d_means;
  std::vector<thrust::device_vector<float> > d_transforms;
  std::vector<thrust::device_vector<float> > d_reconstructed;

	std::vector<thrust::host_vector<float> > h_t;
	std::vector<thrust::device_vector<float> > d_t;

  std::vector<thrust::host_vector<float> > h_pca_bases;
  std::vector<thrust::device_vector<float> > d_pca_bases;
};