#include <cublas_v2.h>
#include "head.h"

//A(M*K) B(K*N) C(M*N) -> C = alpha * A * B + beta * C
void matrixMultiply(cublasHandle_t &handle, thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, uint M_, uint N_, uint K_, float alpha, float beta);
void matrixMultiply(cublasHandle_t &handle, float* &A, float* &B, float* &C, uint M_, uint N_, uint K_, float alpha, float beta);

void replicateVector(thrust::device_vector<float> &d_result, thrust::device_vector<float> &d_vec, uint N_, uint D_);

void preheat_cublas(uint M_, uint N_, uint K_);