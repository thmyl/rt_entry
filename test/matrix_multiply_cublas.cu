#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// For units of time such as h, min, s, ms, us, ns
using namespace std::literals::chrono_literals;

namespace {
// constexpr size_t M = 400;
// constexpr size_t N = 500;
// constexpr size_t K = 600;
constexpr size_t M = 2;
constexpr size_t N = 4;
constexpr size_t K = 3;
}  // namespace

int main() {
  // Initialize CUBLAS
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "!!!! CUBLAS initialization error\n";
    return EXIT_FAILURE;
  }

  // Initialize the host input matrix;
  std::vector<float> mat_a(M * K, 0.0f);
  std::vector<float> mat_b(K * N, 0.0f);
  std::vector<float> mat_c(M * N, 0.0f);

  // Fill the matrices with random numbers
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 100000);
  auto rand_num = [&dis, &gen]() { return dis(gen); };
  std::generate(mat_a.begin(), mat_a.end(), rand_num);
  std::generate(mat_b.begin(), mat_b.end(), rand_num);
  std::generate(mat_c.begin(), mat_c.end(), rand_num);

  // Show the test matrix data.
  if (M == 2 && N == 4 && K == 3) {
    //      1 2 3
    // A =
    //      4 5 6
    std::iota(mat_a.begin(), mat_a.end(), 1.0f);
    //      1  2  3  4
    // B =  5  6  7  8
    //      9 10 11 12

    std::iota(mat_b.begin(), mat_b.end(), 1.0f);
    std::iota(mat_c.begin(), mat_c.end(), 1.0f);
    //           38 44  50  56
    // C = AB =
    //           83 98 113 128

    std::cout << "A = \n";
    for (size_t i = 0; i < M * K; ++i) {
      std::cout << mat_a[i] << '\t';
      if ((i + 1) % K == 0) {
        std::cout << '\n';
      }
    }
    std::cout << "B = \n";
    for (size_t i = 0; i < K * N; ++i) {
      std::cout << mat_b[i] << '\t';
      if ((i + 1) % N == 0) {
        std::cout << '\n';
      }
    }
    std::cout << "C = \n";
    for(size_t i = 0; i < M * N; ++i){
      std::cout << mat_c[i] << '\t';
      if((i + 1) % N == 0){
        std::cout << '\n';
      }
    }
  }

  // Allocate device memory for the matrices
  float *device_mat_a = nullptr;
  float *device_mat_b = nullptr;
  float *device_mat_c = nullptr;
  if (cudaMalloc(reinterpret_cast<void **>(&device_mat_a),
                 M * K * sizeof(float)) != cudaSuccess) {
    std::cerr << "!!!! device memory allocation error (allocate A)\n";
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&device_mat_b),
                 K * N * sizeof(float)) != cudaSuccess) {
    std::cerr << "!!!! device memory allocation error (allocate B)\n";
    cudaFree(device_mat_a);
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&device_mat_c),
                 M * N * sizeof(float)) != cudaSuccess) {
    std::cerr << "!!!! device memory allocation error (allocate C)\n";
    cudaFree(device_mat_a);
    cudaFree(device_mat_b);
    return EXIT_FAILURE;
  }

  // Initialize the device matrices with the host matrices.
  cudaMemcpy(device_mat_a, mat_a.data(), M * K * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_mat_b, mat_b.data(), K * N * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_mat_c, mat_c.data(), M * N * sizeof(float),
             cudaMemcpyHostToDevice);

  // Free up host memories.
  mat_a.clear();
  mat_a.shrink_to_fit();
  mat_b.clear();
  mat_b.shrink_to_fit();

  // Performs operation using cublas
  // C = alpha * transa(A)*transb(B) + beta * C
  // `transa` indicates whether the matrix A is transposed or not.
  // `transb` indicates whether the matrix B is transposed or not.
  // A: m x k
  // B: k x n
  // C: m x n
  // LDA, LDB, LDC are the leading dimensions of the three matrices,
  // respectively.
  // If C = A x B is calculated, there is alpha = 1.0, beta = 0.0,
  // transa = CUBLAS_OP_N, transb = CUBLAS_OP_N

  // cublasStatus_t cublasSgemm(cublasHandle_t handle,
  //                          cublasOperation_t transa, cublasOperation_t
  //                          transb, int m, int n, int k, const float *alpha,
  //                          const float *A, int lda,
  //                          const float *B, int ldb,
  //                          const float *beta,
  //                          float *C, int ldc);
  float alpha = 2.0f;
  float beta = 1.0f;
  // Note that cublas is column primary! We need to transpose the order!
  // In CPU: C = A x B
  // But in GPU: CT = (A x B)T = BT x AT, T means transpose
  status =
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                  device_mat_b, N, device_mat_a, K, &beta, device_mat_c, N);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "!!!! cublas kernel execution error.\n";
    cudaFree(device_mat_a);
    cudaFree(device_mat_b);
    cudaFree(device_mat_c);
    return EXIT_FAILURE;
  }

  // Read back the result from device memory.
  // std::vector<float> mat_c(M * N, 0.0f);
  cudaMemcpy(mat_c.data(), device_mat_c, M * N * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Show the test results.
  if (M == 2 && N == 4 && K == 3) {
    std::cout << "C = AB = \n";
    for (size_t i = 0; i < M * N; ++i) {
      std::cout << mat_c[i] << '\t';
      if ((i + 1) % N == 0) {
        std::cout << '\n';
      }
    }
  }

  // Memory clean up
  cudaFree(device_mat_a);
  cudaFree(device_mat_b);
  cudaFree(device_mat_c);

  // Shutdown cublas
  cublasDestroy(handle);

  return 0;
}
