#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"
#include "utils/test_utils.h"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

// void GPUfeedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache);
// void GPUbackprop(NeuralNetwork& nn, const arma::mat& y, double reg,
//               const struct cache& bpcache, struct grads& bpgrads);

double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);  // http://arma.sourceforge.net/docs.html#resize_member. Recreate the object according to given size specifications, while preserving the elements as well as the layout of the elements.
    cache.a.resize(2);  // each cache is a std::vector of size 2. 

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    // calculate input to sigmoid. W[i] are the weights of the i^th layer
    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N); // http://arma.sourceforge.net/docs.html#repmat. Generate a matrix by replicating matrix A in a block-like fashion.
    cache.z[0] = z1;

    // calculate first set of activations
    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    // calculate input to sigmoid. 
    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    // calculate second set of activations
    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) { // for each pass through the entire dataset (an epoch) 
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) { // SGD. for each batch of input data. 
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}

// GPU wrapper functions 
void GPUfeedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);  // http://arma.sourceforge.net/docs.html#resize_member. Recreate the object according to given size specifications, while preserving the elements as well as the layout of the elements.
    cache.a.resize(2);  // each cache is a std::vector of size 2. 

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    // CUDA declarations, allocations, memcpy to device
    double* d_z1;
    double* d_z2;
    double *d_a1;
    double* d_a2;
    double* d_W0;
    double *d_W1;
    double* d_X; 
    cudaMalloc((void **)&d_z1, sizeof(double) * nn.b[0].n_rows * N);
    cudaMalloc((void **)&d_z2, sizeof(double) * nn.b[1].n_rows * N);
    cudaMalloc((void **)&d_a1, sizeof(double) * nn.b[0].n_rows * N);
    cudaMalloc((void **)&d_a2, sizeof(double) * nn.b[1].n_rows * N);
    cudaMalloc((void **)&d_W0, sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols);
    cudaMalloc((void **)&d_W1, sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols);
    cudaMalloc((void **)&d_X, sizeof(double) * X.n_rows * X.n_cols);
    cudaMemcpy(d_W0, nn.W[0].memptr(), sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, nn.W[1].memptr(), sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X.memptr(), sizeof(double) * X.n_rows * X.n_cols, cudaMemcpyHostToDevice);

    // calculate input to sigmoid. W[i] are the weights of the i^th layer
    arma::mat z1 = arma::repmat(nn.b[0], 1, N); // initialize output
    cudaMemcpy(d_z1, z1.memptr(), sizeof(double) * nn.b[0].n_rows * N, cudaMemcpyHostToDevice);
    double alpha = 1.0; 
    double beta = 1.0; 
    myGEMM(d_W0, d_X, d_z1, &alpha, &beta, nn.W[0].n_rows, N, X.n_rows);
    cudaMemcpy(z1.memptr(), d_z1, sizeof(double) * nn.W[0].n_rows * N, cudaMemcpyDeviceToHost);
    cache.z[0] = z1;

    // calculate first set of activations
    arma::mat a1(nn.W[0].n_rows, N);
    GPUsigmoid(d_z1, d_a1, nn.W[0].n_rows, N);
    cudaMemcpy(a1.memptr(), d_a1, sizeof(double) * nn.W[0].n_rows * N, cudaMemcpyDeviceToHost);
    cache.a[0] = a1;

    // calculate input to sigmoid. 
    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = arma::repmat(nn.b[1], 1, N); // initialize output. 1 copy per row, N copies per column.
    cudaMemcpy(d_z2, z2.memptr(), sizeof(double) * nn.b[1].n_rows * N, cudaMemcpyHostToDevice);
    myGEMM(d_W1, d_a1, d_z2, &alpha, &beta, nn.W[1].n_rows, z2.n_cols, a1.n_rows);
    cudaMemcpy(z2.memptr(), d_z2, sizeof(double) * nn.W[1].n_rows * N, cudaMemcpyDeviceToHost);
    cache.z[1] = z2;

    // calculate second set of activations
    arma::mat a2(z2.n_rows, z2.n_cols);
    GPUsoftmax(d_z2, d_a2, a2.n_rows, a2.n_cols);
    cudaMemcpy(a2.memptr(), d_a2, sizeof(double) * a2.n_rows * a2.n_cols, cudaMemcpyDeviceToHost);
    cache.a[1] = cache.yc = a2;

    // Cuda deallocation
    cudaFree(d_z1);
    cudaFree(d_z2);
    cudaFree(d_a1);
    cudaFree(d_a2);
    cudaFree(d_W0);
    cudaFree(d_W1);
    cudaFree(d_X);
}

void GPUbackprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int M = y.n_rows; 
    int N = y.n_cols;

    // CUDA declarations, allocations, memcpy to device
    double* d_yc;
    double* d_y;
    double* d_diff; 
    double* d_W1;
    double* d_W0;
    double* d_W1T;
    double* d_db1;
    double* d_da1;
    double* d_dz1_term1;
    double* d_dz1_term2;
    double* d_dz1;
    double* d_db0;
    double* d_a0;
    double* d_a0T;
    double* d_X;
    double* d_XT;
    cudaMalloc((void**)&d_yc, sizeof(double) * M * N);
    cudaMalloc((void**)&d_y, sizeof(double) * M * N);
    cudaMalloc((void**)&d_diff, sizeof(double) * M * N);
    cudaMalloc((void **)&d_W1, sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols);
    cudaMalloc((void **)&d_W0, sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols);
    cudaMalloc((void **)&d_W1T, sizeof(double) * nn.W[1].n_cols * nn.W[1].n_rows);
    cudaMalloc((void **)&d_db1, sizeof(double) * M * 1);
    cudaMalloc((void **)&d_da1, sizeof(double) * nn.W[1].n_cols * N);
    cudaMalloc((void **)&d_dz1_term1, sizeof(double) * nn.W[1].n_cols * N);
    cudaMalloc((void **)&d_dz1_term2, sizeof(double) * nn.W[1].n_cols * N);
    cudaMalloc((void **)&d_dz1, sizeof(double) * nn.W[1].n_cols * N);
    cudaMalloc((void **)&d_db0, sizeof(double) * nn.W[1].n_cols * 1);
    cudaMalloc((void **)&d_a0, sizeof(double) * bpcache.a[0].n_rows * bpcache.a[0].n_cols);
    cudaMalloc((void **)&d_a0T, sizeof(double) * bpcache.a[0].n_cols * bpcache.a[0].n_rows);
    cudaMalloc((void **)&d_X, sizeof(double) * bpcache.X.n_rows * bpcache.X.n_cols);
    cudaMalloc((void **)&d_XT, sizeof(double) * bpcache.X.n_cols * bpcache.X.n_rows);
    cudaMemcpy(d_yc, bpcache.yc.memptr(), sizeof(double) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.memptr(), sizeof(double) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, nn.W[1].memptr(), sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W0, nn.W[0].memptr(), sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, bpcache.X.memptr(), sizeof(double) * bpcache.X.n_rows * bpcache.X.n_cols, cudaMemcpyHostToDevice);

    // yc - y
    GPUaddition(d_yc, d_y, d_diff, 1.0 / N, -1.0 / N, M, N);

    // start calculating gradients
    cudaMemcpy(d_a0, bpcache.a[0].memptr(), sizeof(double) * bpcache.a[0].n_rows * bpcache.a[0].n_cols, cudaMemcpyHostToDevice);
    GPUtranspose(d_a0, d_a0T, bpcache.a[0].n_rows, bpcache.a[0].n_cols);
    double alpha = 1.0;
    myGEMM(d_diff, d_a0T, d_W1, &alpha, &reg, nn.W[1].n_rows, nn.W[1].n_cols, N); 
    // at this point d_W1 holds the result of diff * bpcache.a[0].t() + reg * nn.W[1] which should be assigned to bpgrads.dW[1] 
    arma::mat dW1(nn.W[1].n_rows, nn.W[1].n_cols);
    cudaMemcpy(dW1.memptr(), d_W1, sizeof(double) * dW1.n_rows * dW1.n_cols, cudaMemcpyDeviceToHost);
    bpgrads.dW[1] = dW1;

    // calculate db1 
    arma::mat db1(M, 1);
    GPUsum(d_diff, d_db1, M, N, 1);
    cudaMemcpy(db1.memptr(), d_db1, sizeof(double) * M * 1, cudaMemcpyDeviceToHost);
    bpgrads.db[1] = db1;

    // calculate transpose of nn.W[1] to calculate da1 
    cudaMemcpy(d_W1, nn.W[1].memptr(), sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyHostToDevice); // TODO: optimize this later. only needed since we overwrote d_W1
    GPUtranspose(d_W1, d_W1T, nn.W[1].n_rows, nn.W[1].n_cols);

    // compute da1
    double beta = 0.0;
    myGEMM(d_W1T, d_diff, d_da1, &alpha, &beta, nn.W[1].n_cols, N, nn.W[1].n_rows); 

    // now calculate dz1
    GPUelemwise(d_da1, d_a0, d_dz1_term1, nn.W[1].n_cols, N);
    GPUelemwise(d_dz1_term1, d_a0, d_dz1_term2, nn.W[1].n_cols, N);
    GPUaddition(d_dz1_term1, d_dz1_term2, d_dz1, 1.0, -1.0, nn.W[1].n_cols, N);

    // calculate dw0
    arma::mat dW0(nn.W[0].n_rows, nn.W[0].n_cols);
    GPUtranspose(d_X, d_XT, bpcache.X.n_rows, bpcache.X.n_cols);
    myGEMM(d_dz1, d_XT, d_W0, &alpha, &reg, nn.W[0].n_rows, nn.W[0].n_cols, N);
    cudaMemcpy(dW0.memptr(), d_W0, sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyDeviceToHost);
    bpgrads.dW[0] = dW0;

    // calculat db0
    arma::mat db0(nn.W[1].n_cols, 1);
    GPUsum(d_dz1, d_db0, nn.W[1].n_cols, N, 1);
    cudaMemcpy(db0.memptr(), d_db0, sizeof(double) * nn.W[1].n_cols * 1, cudaMemcpyDeviceToHost);
    bpgrads.db[0] = db0;

    // Cuda deallocation 
    cudaFree(d_yc);
    cudaFree(d_y);
    cudaFree(d_diff);
    cudaFree(d_W1);
    cudaFree(d_W0);
    cudaFree(d_W1T);
    cudaFree(d_db1);
    cudaFree(d_da1);
    cudaFree(d_dz1_term1);
    cudaFree(d_dz1_term2);
    cudaFree(d_dz1);
    cudaFree(d_db0);
    cudaFree(d_a0); 
    cudaFree(d_a0T);
    cudaFree(d_X);
    cudaFree(d_XT);
}

/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0)?X.n_cols:0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */

    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */
    int iter = 0;

    // CUDA declarations, allocations, memcpys
    double *d_W0;
    double *d_W1;
    double *d_dW0;
    double *d_dW1;
    double *d_b0;
    double *d_b1;
    double *d_db0;
    double *d_db1;
    cudaMalloc((void **)&d_W0, sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols);
    cudaMalloc((void **)&d_W1, sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols);
    cudaMalloc((void **)&d_dW0, sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols);
    cudaMalloc((void **)&d_dW1, sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols);
    cudaMalloc((void **)&d_b0, sizeof(double) * nn.b[0].n_rows * nn.b[0].n_cols);
    cudaMalloc((void **)&d_b1, sizeof(double) * nn.b[1].n_rows * nn.b[1].n_cols);
    cudaMalloc((void **)&d_db0, sizeof(double) * nn.b[0].n_rows * nn.b[0].n_cols);
    cudaMalloc((void **)&d_db1, sizeof(double) * nn.b[1].n_rows * nn.b[1].n_cols);
    cudaMemcpy(d_W0, nn.W[0].memptr(), sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, nn.W[1].memptr(), sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b0, nn.b[0].memptr(), sizeof(double) * nn.b[0].n_rows * nn.b[0].n_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, nn.b[1].memptr(), sizeof(double) * nn.b[1].n_rows * nn.b[1].n_cols, cudaMemcpyHostToDevice);

    // Gradient descent 
    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */

            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);


            // TODO
            // for all-reduce, we want to transfer each gradient to the host and then all reduce
            // we only all-reduce the delta in the GD calc. so we can transfer that true delta back to each device
            // so everyone can update the network coefficients. then send them back to the host as we do now.

            // Scatter input data to different processes using MPI
            // int num_procs2 = 3; // for debugging 
            assert(X_batch.n_cols == y_batch.n_cols);
            // assert(X_batch.n_rows == y_batch.n_rows); // this is not true 
            int local_size = X_batch.n_cols / num_procs; // approximate number of columns to send to each process (floor)

            // if (rank==0){
            int *displs_x = new int[num_procs];
            int *displs_y = new int[num_procs];
            int *counts_x = new int[num_procs];
            int *counts_y = new int[num_procs];
            // std::cout << "X_batch.n_cols, num_procs, local_size: " << X_batch.n_cols << ", " << num_procs << ", " << local_size << std::endl; 
        

            for (int i = 0; i < num_procs; i++)
            {
                displs_x[i] = X_batch.n_rows*local_size*i;
                displs_y[i] = y_batch.n_rows * local_size * i;
                if (X_batch.n_cols%num_procs != 0 && i == num_procs - 1){ // if the last process needs to be assigned a few extra columns
                    counts_x[i] = X_batch.n_rows * (X_batch.n_cols - local_size * i);
                    counts_y[i] = y_batch.n_rows * (X_batch.n_cols - local_size * i);
                }else{
                    counts_x[i] = X_batch.n_rows * local_size;
                    counts_y[i] = y_batch.n_rows * local_size;
                }
                // std::cout << "X_batch.n_cols, counts[i] and displs[i] are:: " << X_batch.n_cols << ", " << counts[i] / X_batch.n_rows << ", " << displs[i] / X_batch.n_rows << std::endl;
            }
            // }

            // calc how many elements I expect to receive
            int recv_count_x;
            int recv_count_y;
            if (rank==num_procs-1){ // if I'm the last process (who will receive a few extra columns)
                recv_count_x = X_batch.n_rows * (X_batch.n_cols - local_size * rank);
                recv_count_y = y_batch.n_rows * (X_batch.n_cols - local_size * rank);
            }else{
                recv_count_x = X_batch.n_rows * local_size;
                recv_count_y = y_batch.n_rows * local_size;
            }
            arma::mat my_X_batch(X_batch.n_rows, recv_count_x / X_batch.n_rows);
            arma::mat my_y_batch(y_batch.n_rows, recv_count_y / y_batch.n_rows);
            // std::cout << "I expect to receive " << recv_count / X_batch.n_rows << " elements on rank " << rank << std::endl;

            // rank = 0 scatter to other processes. Use scatter_v to handle when the number of processes does not divide evently into total number of columns. 
            MPI_Scatterv(X_batch.memptr(), counts_x, displs_x, MPI_DOUBLE, my_X_batch.memptr(), recv_count_x, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatterv(y_batch.memptr(), counts_y, displs_y, MPI_DOUBLE, my_y_batch.memptr(), recv_count_y, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // forward and backward pass
            struct cache bpcache;
            GPUfeedforward(nn, my_X_batch, bpcache); 

            struct grads bpgrads;
            GPUbackprop(nn, my_y_batch, reg, bpcache, bpgrads);

            // Allocate memory for global gradients
            double *dW0 = new double[bpgrads.dW[0].n_rows * bpgrads.dW[0].n_cols];
            double *dW1 = new double[bpgrads.dW[1].n_rows * bpgrads.dW[1].n_cols];
            double *db0 = new double[bpgrads.db[0].n_rows * bpgrads.db[0].n_cols];
            double *db1 = new double[bpgrads.db[1].n_rows * bpgrads.db[1].n_cols];

            // MPI all reduce local bpgrads
            MPI_Allreduce(bpgrads.dW[0].memptr(), dW0, bpgrads.dW[0].n_rows * bpgrads.dW[0].n_cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(bpgrads.dW[1].memptr(), dW1, bpgrads.dW[1].n_rows * bpgrads.dW[1].n_cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(bpgrads.db[0].memptr(), db0, bpgrads.db[0].n_rows * bpgrads.db[0].n_cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(bpgrads.db[1].memptr(), db1, bpgrads.db[1].n_rows * bpgrads.db[1].n_cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // transfer reduced bpgrads to each gpu
            cudaMemcpy(d_dW0, dW0, sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dW1, dW1, sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyHostToDevice);
            cudaMemcpy(d_db0, db0, sizeof(double) * nn.b[0].n_rows * nn.b[0].n_cols, cudaMemcpyHostToDevice);
            cudaMemcpy(d_db1, db1, sizeof(double) * nn.b[1].n_rows * nn.b[1].n_cols, cudaMemcpyHostToDevice);

            // Gradient descent - W0 
            GPUaddition(d_W0, d_dW0, d_W0, 1.0, -learning_rate, nn.W[0].n_rows, nn.W[0].n_cols);
            arma::mat W0(nn.W[0].n_rows, nn.W[0].n_cols);
            cudaMemcpy(W0.memptr(), d_W0, sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyDeviceToHost);
            nn.W[0] = W0;

            // Gradient descent - W1
            GPUaddition(d_W1, d_dW1, d_W1, 1.0, -learning_rate, nn.W[1].n_rows, nn.W[1].n_cols);
            arma::mat W1(nn.W[1].n_rows, nn.W[1].n_cols);
            cudaMemcpy(W1.memptr(), d_W1, sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyDeviceToHost);
            nn.W[1] = W1;

            // Gradient descent - b0
            GPUaddition(d_b0, d_db0, d_b0, 1.0, -learning_rate, nn.b[0].n_rows, nn.b[0].n_cols);
            arma::mat b0(nn.b[0].n_rows, nn.b[0].n_cols);
            cudaMemcpy(b0.memptr(), d_b0, sizeof(double) * nn.b[0].n_rows * nn.b[0].n_cols, cudaMemcpyDeviceToHost);
            nn.b[0] = b0;

            // Gradient descent - b1
            GPUaddition(d_b1, d_db1, d_b1, 1.0, -learning_rate, nn.b[1].n_rows, nn.b[1].n_cols);
            arma::mat b1(nn.b[1].n_rows, nn.b[1].n_cols);
            cudaMemcpy(b1.memptr(), d_b1, sizeof(double) * nn.b[1].n_rows * nn.b[1].n_cols, cudaMemcpyDeviceToHost);
            nn.b[1] = b1;


            // do not make any edits past here. All of this should be fine. 
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }

    error_file.close();

    // CUDA deallocation
    cudaFree(d_W0);
    cudaFree(d_W1);
    cudaFree(d_dW0);
    cudaFree(d_dW1);
    cudaFree(d_b0);
    cudaFree(d_b1);
    cudaFree(d_db0);
    cudaFree(d_db1);
}
