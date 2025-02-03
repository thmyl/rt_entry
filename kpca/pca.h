#pragma once

#include <Dense>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
// #include "algo.h"

class PCA {
public:
  PCA();
  PCA(float *datasets, uint _nb, uint _dim);
  
  void reset(const Eigen::MatrixXd& data, uint _dim, std::vector<uint>& idlist);
  void pca_transform(const Eigen::MatrixXd &data, Eigen::MatrixXd &res, uint pj_dim);
  void pca_inverse_transform(const Eigen::MatrixXd &data, Eigen::MatrixXd &res, uint pj_dim);

  void calc_eigenvalues();
  void save_mean_rotation(const char *mean_path, const char *rotation_path);
  void read_mean_rotation(const char *mean_path, const char *rotation_path);
  void calc_result(uint pj_dim);
  void save_result(uint pj_dim, const char *pca_base_path);

  void computeCov(Eigen::MatrixXd &, Eigen::MatrixXd &);
  void computeEig(Eigen::MatrixXd &, Eigen::MatrixXd &, Eigen::RowVectorXd &);
  double Ratio(uint);

  Eigen::MatrixXd B;
  Eigen::MatrixXd C;//协方差矩阵
  Eigen::MatrixXd B_res;//投影后的数据
  Eigen::MatrixXd vec; //特征向量(前i列对应前i个特征)
  Eigen::RowVectorXd val; //特征值
  Eigen::RowVectorXd meanvecRow; //均值向量
  uint nb, dim; // dim投影前维度
}; // end of class PCA