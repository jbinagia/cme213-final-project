#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"
#include <armadillo>
#include "utils/common.h"

__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
Just a dummy function that can be used to warm up GPU
*/
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_result));
    return result;
}

__global__
void gpuGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double alpha, double beta,
           int M, int N, int K) {

    uint i = (blockIdx.y * blockDim.y) + threadIdx.y; // let this correspond to row index
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; // let this correspond to column index 

    if (i < M && j < N){ 
        C[j*M + i] = beta*C[j*M + i]; 
        for (int k = 0; k < K; k++){
            C[j*M + i] += alpha*A[k*M + i]*B[j*K + k]; 
        }
    }
}

__global__
void sigmoidKernel(double* mat1, double* mat2, int M, int N) {

    uint i = (blockIdx.y * blockDim.y) + threadIdx.y; // let this correspond to row index
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; // let this correspond to column index 

    if (i < M && j < N){ 
        mat2[j*M + i] = 1.0/(1.0 + exp(-mat1[j*M + i])); 
    }
}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */

    // here is where I need to implement the CUDA GEMM kernel
    // - need to dereference alpha and beta via *alpha before using them 
    // - A, B, C are all device arrays at this point 

    dim3 threadsPerBlock(8, 32);  // 256 threads
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; // N is number of columns
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y; // M is number of rows
    dim3 numBlocks(num_blocks_x, num_blocks_y); 
    gpuGEMM<<<numBlocks, threadsPerBlock>>>(A, B, C, *alpha, *beta, M, N, K); 

    return 1;
}

// wrapper to automate allocating and transferring memory 
int wrapperGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K) {

    double* dA;
    double* dB;
    double* dC;

    cudaMalloc((void**)&dA, sizeof(double) * M * K);
    cudaMalloc((void**)&dB, sizeof(double) * K * N);
    cudaMalloc((void**)&dC, sizeof(double) * M * N);

    cudaMemcpy(dA, A, sizeof(double) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(double) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(double) * M * N, cudaMemcpyHostToDevice);

    int err = myGEMM(dA, dB, dC, alpha, beta, M, N, K); 
    cudaMemcpy(C, dC, sizeof(double) * M * N, cudaMemcpyDeviceToHost);

    return 1;
}

void GPUsigmoid(const arma::mat& mat1, arma::mat& mat2) {

    int M = mat1.n_rows; 
    int N = mat1.n_cols; 

    mat2.set_size(M, N);
    ASSERT_MAT_SAME_SIZE(mat1, mat2);

    dim3 threadsPerBlock(8, 32);  // 256 threads
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; // N is number of columns
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y; // M is number of rows
    dim3 numBlocks(num_blocks_x, num_blocks_y); 

    double* dmat1;
    double* dmat2;

    cudaMalloc((void**)&dmat1, sizeof(double) * M * N);
    cudaMalloc((void**)&dmat2, sizeof(double) * M * N);

    cudaMemcpy(dmat1, mat1.memptr(), sizeof(double) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dmat2, mat2.memptr(), sizeof(double) * M * N, cudaMemcpyHostToDevice);

    sigmoidKernel<<<numBlocks, threadsPerBlock>>>(dmat1, dmat2, M, N); 

    cudaMemcpy(mat2.memptr(), dmat2, sizeof(double) * M * N, cudaMemcpyDeviceToHost);
}

// void softmax(const arma::mat& mat, arma::mat& mat2) {
//     mat2.set_size(mat.n_rows, mat.n_cols);
//     arma::mat exp_mat = arma::exp(mat);
//     arma::mat sum_exp_mat = arma::sum(exp_mat, 0);
//     mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
// }

