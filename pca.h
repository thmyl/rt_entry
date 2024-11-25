#pragma once

#include "Eigen/Dense"
#include <bits/stdc++.h>
#include <cuda_runtime.h>
// #include "algo.h"

class PCA {
public:
  PCA() {};
  PCA(float *datasets, uint _nb, uint _dim) {
    nb = _nb;
    dim = _dim;
    printf("nb = %d, d = %d\n", nb, dim);
    B.resize(nb, dim);
    C.resize(dim, dim);
    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < dim; j++) {
        B(i, j) = datasets[i * dim + j];
      }
    }
  }

  void calc_eigenvalues();
  void save_mean_rotation(const char *mean_path, const char *rotation_path);
  void read_mean_rotation(const char *mean_path, const char *rotation_path);
  void calc_result(uint pj_dim);
  void save_result(uint pj_dim, const char *pca_base_path);

  void computeCov(Eigen::MatrixXd &, Eigen::MatrixXd &);
  void computeEig(Eigen::MatrixXd &, Eigen::MatrixXd &, Eigen::MatrixXd &);
  double Ratio(uint);

  Eigen::MatrixXd B, C, B_res;
  Eigen::MatrixXd vec, val;
  Eigen::RowVectorXd meanvecRow;
  uint nb, dim; // dim投影前维度
}; // end of class PCA

void subtraction(float* A, float* B, uint nq, uint dim_);//A的每一行减B
void rotate(float3* C, float* A, float* B, uint nq, uint dim_);//A乘以B的left到left+3列