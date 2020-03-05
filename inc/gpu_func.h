#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <armadillo>

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int useless_gpu_add_one(int t);

__global__
void gpuGEMM(double* A, double* B, double* C, double alpha, double beta, int M,
           int N, int K);

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K);

int wrapperGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K);

void GPUsigmoid(const arma::mat& mat, arma::mat& mat2);

void GPUsoftmax(double* mat, double* mat2, int M, int N); 

void GPUaddition(double* mat, double* mat2, double* output_mat, double alpha, double beta, int M, int N);

void GPUscalar_mult(double scalar, double* mat, int M, int N);

__global__
void sigmoidKernel(double* mat1, double* mat2, int M, int N);

__global__
void exponentialKernel(double* mat1, double* mat2, int M, int N);

__global__
void softmaxKernel(double* mat1, double* mat2, int M, int N);

__global__
void sumcols(double* mat1, double* mat2, int M, int N);

__global__
void repmat(double* mat1, double* mat2, int M, int N);

__global__
void addmat(double* mat1, double* mat2, double* output_mat, int M, int N);


#endif
