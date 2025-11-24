#include <cublas_v2.h>
#include "head.h"

//A(M*K) B(K*N) C(M*N) -> C = alpha * A * B + beta * C
void matrixMultiply(cublasHandle_t &handle, thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, uint M_, uint N_, uint K_, float alpha, float beta);
void matrixMultiply(cublasHandle_t &handle, float* &A, float* &B, float* &C, uint M_, uint N_, uint K_, float alpha, float beta);

//A(M*K) B(N*K) C(M*N) -> C = alpha * A * B^T + beta * C
void matrixMultiplyABT(cublasHandle_t &handle, thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, uint M_, uint N_, uint K_, float alpha, float beta);
void matrixMultiplyABT(cublasHandle_t &handle, float* &A, float* &B, float* &C, uint M_, uint N_, uint K_, float alpha, float beta);

void replicateVector(float* d_result, float* d_vec, uint N_, uint D_);

void preheat_cublas(uint M_, uint N_, uint K_);

void computeRowNorms(const float* data, float* norms, int rows, int cols);
void addNormsToDistances(float* distances, const float* row_norms, const float* col_norms, int rows, int cols);