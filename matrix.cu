#include "pca.h"
#include "matrix.h"

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

void matrixMultiply(cublasHandle_t handle, thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, uint M_, uint N_, uint K_, float alpha, float beta){
  
  auto *A_ptr = thrust::raw_pointer_cast(A.data());
  auto *B_ptr = thrust::raw_pointer_cast(B.data());
  auto *C_ptr = thrust::raw_pointer_cast(C.data());
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_, M_, K_, &alpha, B_ptr, N_, A_ptr, K_, &beta, C_ptr, N_);
  // cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_, M_, K_, &alpha, B_ptr, N_, A_ptr, K_, &beta, C_ptr, N_);
  // if(status != CUBLAS_STATUS_SUCCESS){
  //   std::cerr << "!!!! CUBLAS kernel execution error.\n";
  // }
  
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