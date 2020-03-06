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

__global__
void exponentialKernel(double* mat1, double* mat2, int M, int N) {

    uint i = (blockIdx.y * blockDim.y) + threadIdx.y; // let this correspond to row index
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; // let this correspond to column index 

    if (i < M && j < N){ 
        mat2[j*M + i] = exp(mat1[j*M + i]); 
    }
}

__global__
void softmaxKernel(double* mat1, double* mat2, int M, int N) {

    uint i = (blockIdx.y * blockDim.y) + threadIdx.y; // let this correspond to row index
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; // let this correspond to column index 

    if (i < M && j < N){ 
        mat2[j*M + i] = mat2[j*M+i]/mat1[j*M+i]; 
    }
}

__global__
void sum(double* mat1, double* mat2, int M, int N, int dim) {
    // dim = 0: calc sum for each column 
    // dim = 1: calc sum for each row
    if (dim == 0){
        uint j = (blockIdx.x * blockDim.x) + threadIdx.x; // let this correspond to column index 
        if (j < N){ 
            mat2[j] = 0.0; 
            for (int i = 0; i < M; i++){
                    mat2[j] += mat1[j*M + i]; 
            }
        }
    }else if(dim == 1){
        uint i = (blockIdx.x * blockDim.x) + threadIdx.x; // let this correspond to column index 
        if (i < M){ 
            mat2[i] = 0.0; 
            for (int j = 0; j < N; j++){
                    mat2[i] += mat1[j*M + i]; 
            }
        }
    }else{
        printf("Error: dim must be 0 or 1 in sum kernel");
    }
}


__global__
void repmat(double* mat1, double* mat2, int M, int N) {

    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; // let this correspond to column index 
    if (j < N){ 
        for (int i = 0; i < M; i++){
                mat2[j*M + i] = mat1[j]; 
        }
    }
}

__global__
void addmat(double* mat1, double* mat2, double* output_mat, double alpha, double beta, int M, int N) {

    uint i = (blockIdx.y * blockDim.y) + threadIdx.y; // let this correspond to row index
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; // let this correspond to column index 

    if (i < M && j < N){ 
        output_mat[j*M + i] = alpha*mat1[j*M+i] + beta*mat2[j*M+i]; 
    }
}

__global__
void transpose(double* mat, double* output_mat, int M, int N) {

    uint i = (blockIdx.y * blockDim.y) + threadIdx.y; // let this correspond to row index
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; // let this correspond to column index 

    if (i < M && j < N){ 
        output_mat[i*N+j] = mat[j*M + i]; 
    }
}

// kernel for calculating elementwise product between two matrices of size M x N 
__global__
void elemwise(double* mat1, double* mat2, double* output_mat, int M, int N) {

    uint i = (blockIdx.y * blockDim.y) + threadIdx.y; // let this correspond to row index
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; // let this correspond to column index 

    if (i < M && j < N){ 
        output_mat[j*M + i] = mat1[j*M + i]*mat2[j*M + i];  
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

    cudaFree(dmat1);
    cudaFree(dmat2);
}

void GPUsoftmax(double* mat, double* mat2, int M, int N) {

    dim3 threadsPerBlock(8, 32);  // 256 threads
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; // N is number of columns
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y; // M is number of rows
    dim3 numBlocks(num_blocks_x, num_blocks_y); 

    // calculate exponential of each element
    exponentialKernel<<<numBlocks, threadsPerBlock>>>(mat, mat2, M, N); 

    // sum columns of exponential matrix 
    double* d_sum_exp_mat; 
    cudaMalloc((void**)&d_sum_exp_mat, sizeof(double) * 1 * N);
    dim3 threadsPerBlock2(256);
    num_blocks_x = (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x; // N is number of columns
    dim3 numBlocks2(num_blocks_x);
    sum<<<numBlocks2, threadsPerBlock2>>>(mat2, d_sum_exp_mat, M, N, 0);

    // replicate this row vector into a matrix 
    double* d_denom; 
    cudaMalloc((void**)&d_denom, sizeof(double) * M * N);
    repmat<<<numBlocks2, threadsPerBlock2>>>(d_sum_exp_mat, d_denom, M, N); 
    // arma::mat test;
    // test.set_size(1, N);
    // cudaMemcpy(test.memptr(), d_sum_exp_mat, sizeof(double) * 1 * N, cudaMemcpyDeviceToHost);
    // arma::mat arma_denom = repmat(test, M, 1);
    // cudaMemcpy(d_denom, arma_denom.memptr(), sizeof(double) * M * N, cudaMemcpyHostToDevice); // get same error if I do this so replicaiton seems to be okay 


    // finally calculate sigmoid 
    dim3 threadsPerBlock3(8, 32);  // 256 threads
    num_blocks_x = (N + threadsPerBlock3.x - 1)/threadsPerBlock3.x; // N is number of columns
    num_blocks_y = (M + threadsPerBlock3.y - 1)/threadsPerBlock3.y; // M is number of rows
    dim3 numBlocks3(num_blocks_x, num_blocks_y); 
    softmaxKernel<<<numBlocks3, threadsPerBlock3>>>(d_denom, mat2, M, N); 

    // arma::mat exp_mat = arma::exp(mat);
    // arma::mat sum_exp_mat = arma::sum(exp_mat, 0); // For matrix M, return the sum of elements in each column (dim=0), or each row (dim=1)
    // mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1); // Element-wise division of an object by another object or a scalar
}

// operations of the form alpha*A + beta*B
    // e.g. alpha = beta = 1 is simply addition while alpha = 1, beta = -1 is subtraction.
    // or alpha = constant, beta = 0 is multiply scalar by a constant. 
void GPUaddition(double* mat, double* mat2, double* output_mat, double alpha, double beta, int M, int N) {

    dim3 threadsPerBlock(8, 32);  // 256 threads
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; // N is number of columns
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y; // M is number of rows
    dim3 numBlocks(num_blocks_x, num_blocks_y); 

    addmat<<<numBlocks, threadsPerBlock>>>(mat, mat2, output_mat, alpha, beta, M, N); 
}

// calculates sum over rows or column of a matrix of size MxN
void GPUsum(double* mat, double* output_vec, int M, int N, int dim) {

    dim3 threadsPerBlock(256);
    int num_blocks = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; // N is number of columns
    dim3 numBlocks(num_blocks);

    sum<<<numBlocks, threadsPerBlock>>>(mat, output_vec, M, N, dim);
}

// compute transpose of M x N matrix on GPU 
void GPUtranspose(double* __restrict__ mat, double* __restrict__ output_mat, int M, int N) {

    dim3 threadsPerBlock(8, 32);  // 256 threads
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; // N is number of columns
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y; // M is number of rows
    dim3 numBlocks(num_blocks_x, num_blocks_y); 

    transpose<<<numBlocks, threadsPerBlock>>>(mat, output_mat, M, N); 
}

// compute element-wise product of two M x N matrices on the GPU 
void GPUelemwise(double* mat1, double* mat2, double* output_mat, int M, int N) {

    dim3 threadsPerBlock(8, 32);  // 256 threads
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; // N is number of columns
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y; // M is number of rows
    dim3 numBlocks(num_blocks_x, num_blocks_y); 

    elemwise<<<numBlocks, threadsPerBlock>>>(mat1, mat2, output_mat, M, N); 
}

