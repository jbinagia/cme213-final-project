#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"
#include "utils/common.h"

// Thread block size for myGEMM
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 4

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

// Define struct for facilitating computations in gpuGEMM
    // - idea for Matrix, GetElement(), and GetSubMatrix() came from CUDA C++ Programming Guide
    // - Note here matrices are stored in column-major order:
    // - I.e. M(row, col) = *(M.elements + col * M.stride + row)
typedef struct {
    int width;
    int height;
    int stride;
    double* elements;
} Matrix;

// Define function for retrieving element of matrix 
__device__ 
double GetElement(const Matrix A, int row, int col)
{
    return A.elements[col * A.stride + row];
}


// Get the BLOCK_SIZExCsub_ncols sub-matrix Bsub of B that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of B
__device__ 
Matrix GetSubMatrix(Matrix B, int row, int col, int height, int width)
{
    Matrix Bsub;
    Bsub.height = height;
    Bsub.width = width;
    Bsub.stride = B.stride;
    Bsub.elements = &B.elements[B.stride * width * col + height * row];
    return Bsub;
}


__global__
void gpuGEMM(const Matrix A, const Matrix B, Matrix C, double alpha, double beta,
           int M, int N, int K) {

    // Useful definitions
    int linearIdx = threadIdx.x * BLOCK_SIZE_Y + threadIdx.y;  // map each thread to a int between 0 and BLOCK_SIZE_Y*BLOCK_SIZE_X
    int myRowC = BLOCK_SIZE_Y*BLOCK_SIZE_X*blockIdx.y + linearIdx; // what row of A/C this thread is responsible for 

    // Determine Csub_ncols, i.e. number of columns of Bsub we will actually utilize
    int nom_last_col_B = BLOCK_SIZE_X*(blockIdx.x + 1) - 1; // last column of B we will nominally consider
    int Csub_ncols = nom_last_col_B <= N-1 ? BLOCK_SIZE_X : (N - BLOCK_SIZE_X*(blockIdx.x)); // this prevents column index of C/B from going out of bounds

    // Shared memory used to store Bsub 
    __shared__ double Bs[BLOCK_SIZE_Y][BLOCK_SIZE_X]; // 2d array 

    // for each iteration, load l-th 4xCsub_ncols chunk of B into shared memory 
    int num_iters = K%BLOCK_SIZE_Y==0 ? K/BLOCK_SIZE_Y : K/BLOCK_SIZE_Y + 1; // if number of rows of B isn't divisible by K, we need to do one extra iteration with less rows of B 
    for (int l = 0; l < num_iters; l++){
        // each thread multiplies 1xBLOCK_SIZE_Y chunk of Asub with BLOCK_SIZE_Yx Csub_ncols Bsub (which is in shared memory) to 
        // compute contribution to a 1xCsub_ncols chunk of Csub

        // Calculate number of rows of B to actually read from. Bsub_nrows < BLOCK_SIZE_Y if K not divisible by BLOCK_SIZE_Y and on final iteration 
        int Bsub_nrows =  (K%BLOCK_SIZE_Y!=0 && l==num_iters-1) ? (K - BLOCK_SIZE_Y*l) :  BLOCK_SIZE_Y;  

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, l, blockIdx.x, BLOCK_SIZE_Y, BLOCK_SIZE_X);

        // Thread row and column within Bsub
        int row = threadIdx.y;
        int col = threadIdx.x;

        // Now each thread retrieves a 1 x BLOCK_SIZE_Y chunk of A (we only need 1 x Bsub_nrows of it)
        double a[BLOCK_SIZE_Y]; // local 1 x BLOCK_SIZE_Y array 

        // prevent row index of B / column index of A from going out of bounds
        int row_in_B = row + l*BLOCK_SIZE_Y; 
        int column_in_B = col + blockIdx.x*BLOCK_SIZE_X; // this prevents column index of C/B from going out of bounds
        if (row_in_B < K && column_in_B < N){

            // Load Bsub from device memory to shared memory
            // Each thread loads one element of each sub-matrix
            Bs[row][col] = GetElement(Bsub, row, col);
        }

        // Get sub-matrix Bsub of B
        Matrix Asub = GetSubMatrix(A, myRowC, l, 1, BLOCK_SIZE_Y);

        // fill up local array a 
        for (int m = 0; m < Bsub_nrows; m++){
            a[m] = GetElement(Asub,0,m);
        }
        
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // prevent row index of A/C from going out of bounds 
        if (myRowC < M){

            // Each thread computes one 1xBLOCK_SIZE_X sub-matrix Csub of C
            Matrix Csub = GetSubMatrix(C, myRowC, blockIdx.x, 1, BLOCK_SIZE_X);

            // Multiply a and Bsub together
            for (int k = 0; k < Csub_ncols; k++){ // for each column in Csub. this prevents column index of C/B from going out of bounds

                // for each column, we are basically accumulating into the k-th column of my Csub by adding the product of (local a)*(Bs[:,k])
                for (int m =0; m < Bsub_nrows; m++){ // for each element of A, row of Bs
                    if (m==0 && l==0){  // if this is our first pass multiply what was stored in C by beta
                        Csub.elements[k * Csub.stride] = beta*Csub.elements[k * Csub.stride] + alpha*a[m]*Bs[m][k]; 
                    }else{      // simply accumulate
                        Csub.elements[k * Csub.stride] += alpha*a[m]*Bs[m][k];
                    } 
                }
            }
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
}

__global__
void naiveGEMM(double* __restrict__ A, double* __restrict__ B,
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
    This version is the naive implementation that I implemented first. 
*/
int myNaiveGEMM(double* __restrict__ A, double* __restrict__ B,
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
    naiveGEMM<<<numBlocks, threadsPerBlock>>>(A, B, C, *alpha, *beta, M, N, K); 

    return 0;
}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
    This version follows that described by section 4.2 in the final project handout 1. 
*/
int myGEMM(double* A, double* B,
           double* C, double* alpha, double* beta,
           int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */

    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);  // 64 threads per block 
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; // N is number of columns of C
    int num_blocks_y = (M + threadsPerBlock.x*threadsPerBlock.y - 1)/threadsPerBlock.x/threadsPerBlock.y; // M is number of rows of C
    // std::cout << num_blocks_x << " and " << num_blocks_y << std::endl; // makes sense for simple cases now 
    dim3 numBlocks(num_blocks_x, num_blocks_y); 

    // Define A, B, C as matrices to faciliate computation
    Matrix d_A, d_B, d_C;
    d_A.height = d_A.stride = M; d_A.width = K; d_A.elements = A; 
    d_B.height = d_B.stride = K; d_B.width = N; d_B.elements = B; 
    d_C.height = d_C.stride = M; d_C.width = N; d_C.elements = C; 
    gpuGEMM<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, *alpha, *beta, M, N, K); 

    return 0;
}

void GPUsigmoid(double* mat1, double* mat2, int M, int N) {

    dim3 threadsPerBlock(8, 32);  // 256 threads
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; // N is number of columns
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y; // M is number of rows
    dim3 numBlocks(num_blocks_x, num_blocks_y); 

    sigmoidKernel<<<numBlocks, threadsPerBlock>>>(mat1, mat2, M, N); 
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

    // finally calculate sigmoid 
    dim3 threadsPerBlock3(8, 32);  // 256 threads
    num_blocks_x = (N + threadsPerBlock3.x - 1)/threadsPerBlock3.x; // N is number of columns
    num_blocks_y = (M + threadsPerBlock3.y - 1)/threadsPerBlock3.y; // M is number of rows
    dim3 numBlocks3(num_blocks_x, num_blocks_y); 
    softmaxKernel<<<numBlocks3, threadsPerBlock3>>>(d_denom, mat2, M, N); 
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

