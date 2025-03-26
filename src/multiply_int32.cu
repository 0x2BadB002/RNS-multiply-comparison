#include "multiply_int32.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void matrixMultiplyKernel(const int32_t *A, const int32_t *B,
                                     int32_t *C, size_t N) {
  int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  int32_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    int32_t sum = 0;
    for (int32_t k = 0; k < N; ++k) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

void matrixMultiplyCUDA(const int32_t *A, const int32_t *B, int32_t *C,
                        size_t N) {
  int32_t *d_A, *d_B, *d_C;
  size_t size = N * N * sizeof(int32_t);

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                (N + blockSize.y - 1) / blockSize.y);

  matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// RNS
const int moduli[] = {2, 3, 5, 7, 11, 13, 17, 19};
const int num_moduli = sizeof(moduli) / sizeof(moduli[0]);
const int M = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19;

// Precomputed (M_i * inv_i mod M) (See misc/moduli.py)
const int term_i[] = {4849845, 3233230, 3879876, 8314020,
                      6172530, 3730650, 9129120, 9189180};

__global__ void convertToRNSKernel(const int32_t *A, int8_t *A_res,
                                   const int *d_moduli, size_t N) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= N || y >= N)
    return;

  size_t idx = y * N + x;
  int32_t val = A[idx];

  for (int i = 0; i < num_moduli; ++i) {
    int m = d_moduli[i];
    int residue = val % m;
    if (residue < 0)
      residue += m;
    A_res[i * N * N + idx] = static_cast<int8_t>(residue);
  }
}

__global__ void matrixMulModKernel(const int8_t *A, const int8_t *B, int8_t *C,
                                   int m, size_t N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= N || col >= N)
    return;

  int8_t sum = 0;
  for (size_t k = 0; k < N; ++k) {
    int a = A[row * N + k];
    int b = B[k * N + col];

    sum = (sum + (a * b) % m) % m;
  }

  if (sum < 0)
    sum += m;

  C[row * N + col] = static_cast<int8_t>(sum);
}

__global__ void combineCRTKernel(const int8_t *C_res, int32_t *C,
                                 const int *d_term_i, size_t N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= N || col >= N)
    return;

  size_t idx = row * N + col;
  int32_t sum = 0;
  for (int i = 0; i < num_moduli; ++i) {
    int8_t c_i = C_res[i * N * N + idx];
    sum += c_i * d_term_i[i];
  }
  sum %= M;
  if (sum > M / 2)
    sum -= M;
  C[idx] = sum;
}

void rnsMatrixMultiply(const int32_t *h_A, const int32_t *h_B, int32_t *h_C,
                       size_t N) {
  int32_t *d_A, *d_B;
  int8_t *d_A_res, *d_B_res, *d_C_res;
  int32_t *d_C;
  int *d_moduli, *d_term_i;

  cudaMalloc(&d_A, N * N * sizeof(int32_t));
  cudaMalloc(&d_B, N * N * sizeof(int32_t));
  cudaMemcpy(d_A, h_A, N * N * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N * N * sizeof(int32_t), cudaMemcpyHostToDevice);

  size_t rns_size = num_moduli * N * N * sizeof(int8_t);
  cudaMalloc(&d_A_res, rns_size);
  cudaMalloc(&d_B_res, rns_size);

  cudaMalloc(&d_moduli, num_moduli * sizeof(int));
  cudaMemcpy(d_moduli, moduli, num_moduli * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMalloc(&d_term_i, num_moduli * sizeof(int));
  cudaMemcpy(d_term_i, term_i, num_moduli * sizeof(int),
             cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (N + 15) / 16);
  convertToRNSKernel<<<grid, block>>>(d_A, d_A_res, d_moduli, N);
  convertToRNSKernel<<<grid, block>>>(d_B, d_B_res, d_moduli, N);
  cudaDeviceSynchronize();

  cudaMalloc(&d_C_res, rns_size);

  for (int i = 0; i < num_moduli; ++i) {
    matrixMulModKernel<<<grid, block>>>(d_A_res + i * N * N,
                                        d_B_res + i * N * N,
                                        d_C_res + i * N * N, moduli[i], N);
  }
  cudaDeviceSynchronize();

  cudaMalloc(&d_C, N * N * sizeof(int32_t));
  combineCRTKernel<<<grid, block>>>(d_C_res, d_C, d_term_i, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, N * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_A_res);
  cudaFree(d_B_res);
  cudaFree(d_C_res);
  cudaFree(d_C);
  cudaFree(d_moduli);
  cudaFree(d_term_i);
}
