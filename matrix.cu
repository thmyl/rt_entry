#include "pca.h"
#include "matrix.h"
#include <sys/time.h>
#include <cuda_profiler_api.h>

extern void check_gpu_memory();

__global__ void substraction_kernel(float* A, float* B, uint nq, uint dim_){
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < nq){
    for(int i = 0; i < dim_; i++){
      A[tid*dim_ + i] -= B[i];
    }
  }
}

void subtraction(float* A, float* B, uint nq, uint dim_){
  substraction_kernel<<<(nq+255)/256, 256>>>(A, B, nq, dim_);
  CUDA_SYNC_CHECK();
}

__global__ void rotate_kernel(float3* C, float* A, float* B, uint nq, uint dim_){
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < nq){
    C[tid] = make_float3(0, 0, 0);
    for(int i=0; i<dim_; i++){
      C[tid].x += A[tid*dim_ + i] * B[i*3 + 0];
      C[tid].y += A[tid*dim_ + i] * B[i*3 + 1];
      C[tid].z += A[tid*dim_ + i] * B[i*3 + 2];
    }
  }
}

void rotate(float3* C, float* A, float* B, uint nq, uint dim_){
  rotate_kernel<<<(nq+255)/256, 256>>>(C, A, B, nq, dim_);
  CUDA_SYNC_CHECK();
}

void matrixMultiply(cublasHandle_t &handle, thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, uint M_, uint N_, uint K_, float alpha, float beta){
  #ifdef DETAIL
    printf("M_ = %d, N_ = %d, K_ = %d\n", M_, N_, K_);
  #endif
  auto *A_ptr = thrust::raw_pointer_cast(A.data());
  auto *B_ptr = thrust::raw_pointer_cast(B.data());
  auto *C_ptr = thrust::raw_pointer_cast(C.data());
  // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_, M_, K_, &alpha, B_ptr, N_, A_ptr, K_, &beta, C_ptr, N_);
  // Timing::startTiming("matrix multiply");
  cublasGemmEx(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N_,
                M_,
                K_,
                &alpha,
                B_ptr,
                CUDA_R_32F,
                N_,
                A_ptr,
                CUDA_R_32F,
                K_,
                &beta,
                C_ptr,
                CUDA_R_32F,
                N_,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  // Timing::stopTiming(2);
}

void matrixMultiply(cublasHandle_t &handle, float* &A, float* &B, float* &C, uint M_, uint N_, uint K_, float alpha, float beta){
  
  // Timing::startTiming("matrix multiply");
  printf("N_ = %d, M_ = %d, K_ = %d\n", N_, M_, K_);
  cublasGemmEx(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N_,
                M_,
                K_,
                &alpha,
                B,
                CUDA_R_32F,
                N_,
                A,
                CUDA_R_32F,
                K_,
                &beta,
                C,
                CUDA_R_32F,
                N_,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  // Timing::stopTiming(2);
}

void matrixMultiplyABT(cublasHandle_t &handle, thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, uint M_, uint N_, uint K_, float alpha, float beta){
  #ifdef DETAIL
    printf("M_ = %d, N_ = %d, K_ = %d\n", M_, N_, K_);
  #endif
  auto *A_ptr = thrust::raw_pointer_cast(A.data());
  auto *B_ptr = thrust::raw_pointer_cast(B.data());
  auto *C_ptr = thrust::raw_pointer_cast(C.data());
  // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_, M_, K_, &alpha, B_ptr, N_, A_ptr, K_, &beta, C_ptr, N_);
  // Timing::startTiming("matrix multiply");
  // cublasGemmEx(handle,
  //               CUBLAS_OP_N,
  //               CUBLAS_OP_T,
  //               N_,
  //               M_,
  //               K_,
  //               &alpha,
  //               B_ptr, CUDA_R_32F, K_,
  //               A_ptr, CUDA_R_32F, K_,
  //               &beta,
  //               C_ptr, CUDA_R_32F, N_,
  //               CUDA_R_32F,
  //               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasGemmEx(handle,
    CUBLAS_OP_T,
    CUBLAS_OP_N,
    N_,
    M_,
    K_,
    &alpha,
    B_ptr, CUDA_R_32F, K_,
    A_ptr, CUDA_R_32F, K_,
    &beta,
    C_ptr, CUDA_R_32F, N_,
    CUDA_R_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  // Timing::stopTiming(2);
}

void matrixMultiplyABT(cublasHandle_t &handle, float* &A, float* &B, float* &C, uint M_, uint N_, uint K_, float alpha, float beta){
  
  // Timing::startTiming("matrix multiply");
  printf("N_ = %d, M_ = %d, K_ = %d\n", N_, M_, K_);
  cublasGemmEx(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                N_,
                M_,
                K_,
                &alpha,
                B, CUDA_R_32F, K_,
                A, CUDA_R_32F, K_,
                &beta,
                C, CUDA_R_32F, N_,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  // Timing::stopTiming(2);
}

void preheat_cublas(uint M_, uint N_, uint K_){
  #ifdef DETAIL
    printf("pre cublas\n");
    Timing::startTiming("pre cublas");
  #endif
  cublasHandle_t handle;
  cublasCreate(&handle);
  float* A;
  float* B;
  float* C;
  cudaMalloc(&A, M_ * K_ * sizeof(float));
  cudaMalloc(&B, K_ * N_ * sizeof(float));
  cudaMalloc(&C, M_ * N_ * sizeof(float));
  float alpha = 1.0;
  float beta = 0.0;
  cublasGemmEx(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N_,
                M_,
                K_,
                &alpha,
                B,
                CUDA_R_32F,
                N_,
                A,
                CUDA_R_32F,
                K_,
                &beta,
                C,
                CUDA_R_32F,
                N_,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  #ifdef DETAIL
    Timing::stopTiming();
  #endif
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cublasDestroy(handle);
}

__global__ void row_norm_kernel(const float* data, float* norms, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const float* row_ptr = data + static_cast<size_t>(row) * cols;
  float sum = 0.0f;
  for (int col = 0; col < cols; ++col) {
    float v = row_ptr[col];
    sum += v * v;
  }
  norms[row] = sum;
}

void computeRowNorms(const float* data, float* norms, int rows, int cols) {
  int block = 256;
  int grid = (rows + block - 1) / block;
  row_norm_kernel<<<grid, block>>>(data, norms, rows, cols);
  CUDA_SYNC_CHECK();
}

__global__ void add_norms_kernel(float* distances,
                                 const float* row_norms,
                                 const float* col_norms,
                                 int rows,
                                 int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx >= total) return;
  int row = idx / cols;
  int col = idx % cols;
  distances[idx] += row_norms[row] + col_norms[col];
}

void addNormsToDistances(float* distances,
                         const float* row_norms,
                         const float* col_norms,
                         int rows,
                         int cols) {
  int block = 256;
  int grid = (rows * cols + block - 1) / block;
  add_norms_kernel<<<grid, block>>>(distances, row_norms, col_norms, rows, cols);
  CUDA_SYNC_CHECK();
}

__global__ void repeatVector(float* result, float* vec, uint N_, uint D_){
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < N_){
    // if(tid == 0) printf("dim = %d\n", D_);
    for(int i = 0; i < D_; i++){
      result[tid*D_ + i] = vec[i];
    }
  }
}

// void replicateVector(thrust::device_vector<float> &d_result, thrust::device_vector<float> &d_vec, uint N_, uint D_){
//   auto* vec_ptr = thrust::raw_pointer_cast(d_vec.data());
//   auto* result_ptr = thrust::raw_pointer_cast(d_result.data());
void replicateVector(float* result_ptr, float* vec_ptr, uint N_, uint D_){
  repeatVector<<<(N_ + 255)/256, 256>>>(result_ptr, vec_ptr, N_, D_);
  CUDA_SYNC_CHECK();
}