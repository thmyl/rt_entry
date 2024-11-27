#include "pca.h"
#include "matrix.h"
#include <sys/time.h>
#include <cuda_profiler_api.h>

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

void preheat_cublas(uint M_, uint N_, uint K_){
  printf("pre cublas\n");
  Timing::startTiming("pre cublas");
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
  Timing::stopTiming();
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}

__global__ void repeatVector(float* result, float* vec, uint N_, uint D_){
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < N_){
    if(tid == 0) printf("dim = %d\n", D_);
    for(int i = 0; i < D_; i++){
      result[tid*D_ + i] = vec[i];
    }
  }
}

void replicateVector(thrust::device_vector<float> &d_result, thrust::device_vector<float> &d_vec, uint N_, uint D_){
  auto* vec_ptr = thrust::raw_pointer_cast(d_vec.data());
  auto* result_ptr = thrust::raw_pointer_cast(d_result.data());
  repeatVector<<<(N_ + 255)/256, 256>>>(result_ptr, vec_ptr, N_, D_);
  CUDA_SYNC_CHECK();
}