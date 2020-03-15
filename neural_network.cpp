#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"
#include "utils/test_utils.h"

#define MPI_SAFE_CALL(call)                                         \
    do                                                              \
    {                                                               \
        int err = call;                                             \
        if (err != MPI_SUCCESS)                                     \
        {                                                           \
            fprintf(stderr, "MPI error %d in file '%s' at line %i", \
                    err, __FILE__, __LINE__);                       \
            exit(1);                                                \
        }                                                           \
    } while (0)

// void GPUfeedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache);
// void GPUbackprop(NeuralNetwork& nn, const arma::mat& y, double reg,
//               const struct cache& bpcache, struct grads& bpgrads);

double norms(NeuralNetwork &nn)
{
    double norm_sum = 0;

    for (int i = 0; i < nn.num_layers; ++i)
    {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork &nn, int iter)
{
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

void write_diff_gpu_cpu(NeuralNetwork &nn, int iter,
                        std::ofstream &error_file)
{
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0] - A, "inf") / arma::norm(A, "inf");
    double L2_errW0 = arma::norm(nn.W[0] - A, 2) / arma::norm(A, 2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1] - B, "inf") / arma::norm(B, "inf");
    double L2_errW1 = arma::norm(nn.W[1] - B, 2) / arma::norm(B, 2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0] - C, "inf") / arma::norm(C, "inf");
    double L2_errb0 = arma::norm(nn.b[0] - C, 2) / arma::norm(C, 2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1] - D, "inf") / arma::norm(D, "inf");
    double L2_errb1 = arma::norm(nn.b[1] - D, 2) / arma::norm(D, 2);

    int ow = 15;

    if (iter == 0)
    {
        error_file << std::left << std::setw(ow) << "Iteration" << std::left << std::setw(ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left << std::setw(ow) << "Max Err b0" << std::left << std::setw(ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left << std::setw(ow) << "L2 Err b0" << std::left << std::setw(ow) << "L2 Err b1"
                   << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 << std::left << std::setw(ow) << max_errb0 << std::left << std::setw(ow) << max_errb1 << std::left << std::setw(ow) << L2_errW0 << std::left << std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 << std::left << std::setw(ow) << L2_errb1 << "\n";
}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork &nn, const arma::mat &X, struct cache &cache)
{
    cache.z.resize(2); // http://arma.sourceforge.net/docs.html#resize_member. Recreate the object according to given size specifications, while preserving the elements as well as the layout of the elements.
    cache.a.resize(2); // each cache is a std::vector of size 2.

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
void backprop(NeuralNetwork &nn, const arma::mat &y, double reg,
              const struct cache &bpcache, struct grads &bpgrads)
{
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

void backprop2(NeuralNetwork &nn, const arma::mat &y, double reg,
               const struct cache &bpcache, struct grads &bpgrads, int num_procs)
{
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;
    int normalization = N * num_procs; // we want total number of columns of y here
    // int normalization = 800; // we want total number of columns of y here
    // std::cout << "normalization is: " << normalization << std::endl;

    // std::cout << "backprop " << bpcache.yc << "\n";

    arma::mat diff = (1.0 / (double)normalization) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1]/((double)num_procs);
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0]/((double)num_procs);
    bpgrads.dz1 = dz1;
    bpgrads.Xt = bpcache.X.t();
    bpgrads.w0 = nn.W[0];
    bpgrads.w1 = nn.W[1];
    bpgrads.da1 = da1; 
    bpgrads.diff = diff;
    bpgrads.a0t = bpcache.a[0].t(); 

    bpgrads.db[0] = arma::sum(dz1, 1);

}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork &nn, const arma::mat &yc, const arma::mat &y,
            double reg)
{
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
void predict(NeuralNetwork &nn, const arma::mat &X, arma::rowvec &label)
{
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for (int i = 0; i < X.n_cols; ++i)
    {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork &nn, const arma::mat &X, const arma::mat &y,
             double reg, struct grads &numgrads)
{
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for (int i = 0; i < nn.num_layers; ++i)
    {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for (int j = 0; j < nn.W[i].n_rows; ++j)
        {
            for (int k = 0; k < nn.W[i].n_cols; ++k)
            {
                double oldval = nn.W[i](j, k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2 * h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for (int i = 0; i < nn.num_layers; ++i)
    {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for (int j = 0; j < nn.b[i].size(); ++j)
        {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2 * h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork &nn, const arma::mat &X, const arma::mat &y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug)
{
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for (int epoch = 0; epoch < epochs; ++epoch)
    { // for each pass through the entire dataset (an epoch)
        int num_batches = (N + batch_size - 1) / batch_size;

        for (int batch = 0; batch < num_batches; ++batch)
        { // SGD. for each batch of input data.
            int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if (print_every > 0 && iter % print_every == 0)
            {
                if (grad_check)
                {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" << epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for (int i = 0; i < nn.W.size(); ++i)
            {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for (int i = 0; i < nn.b.size(); ++i)
            {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if (print_every <= 0)
            {
                print_flag = batch == 0;
            }
            else
            {
                print_flag = iter % print_every == 0;
            }

            if (debug && print_flag)
            {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}

// GPU wrapper functions
void GPUfeedforward(NeuralNetwork &nn, const arma::mat &X, struct cache &cache)
{
    cache.z.resize(2); // http://arma.sourceforge.net/docs.html#resize_member. Recreate the object according to given size specifications, while preserving the elements as well as the layout of the elements.
    cache.a.resize(2); // each cache is a std::vector of size 2.

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    // CUDA declarations, allocations, memcpy to device
    double *d_z1;
    double *d_z2;
    double *d_a1;
    double *d_a2;
    double *d_W0;
    double *d_W1;
    double *d_X;
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

void GPUbackprop(NeuralNetwork &nn, const arma::mat &y, double reg,
                 const struct cache &bpcache, struct grads &bpgrads, int num_procs)
{
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int M = y.n_rows;
    int N = y.n_cols;                         // check normalization for yc-y
    int normalization = y.n_cols * num_procs; // TODO: fix for n = 3

    // CUDA declarations, allocations, memcpy to device
    double *d_yc;
    double *d_y;
    double *d_diff;
    double *d_W1;
    double *d_W0;
    double *d_W1T;
    double *d_db1;
    double *d_da1;
    double *d_dz1_term1;
    double *d_dz1_term2;
    double *d_dz1;
    double *d_db0;
    double *d_a0;
    double *d_a0T;
    double *d_X;
    double *d_XT;
    cudaMalloc((void **)&d_yc, sizeof(double) * M * N);
    cudaMalloc((void **)&d_y, sizeof(double) * M * N);
    cudaMalloc((void **)&d_diff, sizeof(double) * M * N);
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
    GPUaddition(d_yc, d_y, d_diff, 1.0 / normalization, -1.0 / normalization, M, N);

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
void parallel_train(NeuralNetwork &nn, const arma::mat &X, const arma::mat &y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug)
{

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0) ? X.n_cols : 0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));
    // std::cout << "Hi from rank " << rank << " my number of columns are " << N << std::endl; // all have same value of N

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


    // -----------------------------------
    // Scatter data to different processes
    // -----------------------------------

    // Broadcast dimensions of X and y to other processes
    int X_n_rows;
    int X_n_cols;
    int y_n_rows;
    int y_n_cols;
    X_n_rows = X.n_rows;
    X_n_cols = X.n_cols;
    y_n_rows = y.n_rows;
    y_n_cols = y.n_cols;
    MPI_Bcast(&X_n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&X_n_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y_n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y_n_cols, 1, MPI_INT, 0, MPI_COMM_WORLD); // can delete all references to y_n_cols 

    // define useful constants 
    int my_num_cols = X_n_cols / num_procs; // approximate number of columns of X and y to distribute to each process (floor)
    int num_batches = (N + batch_size - 1) / batch_size; // number of batches there will be (last batch is of size <= batch_size)
    int minibatch_size = (batch_size + num_procs - 1) / num_procs; // given a batch size, how many training examples to distribute to each process

    // define data structure to hold each process's input data. The i-th entry to these vectors corresponds to batch # i. 
    std::vector<arma::mat> my_X_batches;
    std::vector<arma::mat> my_y_batches;

    // for each batch up to the last batch, scatter the data and place into my_X_batches and my_y_bathes
    arma::mat my_X(X_n_rows, minibatch_size);
    arma::mat my_y(y_n_rows, minibatch_size);
    for (int batch = 0; batch < num_batches-1; ++batch){
        MPI_Scatter(X.memptr(), X_n_rows * minibatch_size, MPI_DOUBLE, my_X.memptr(), X_n_rows * minibatch_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(y.memptr(), y_n_rows * minibatch_size, MPI_DOUBLE, my_y.memptr(), y_n_rows * minibatch_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        my_X_batches.push_back(my_X);
        my_y_batches.push_back(my_y);
    }

    // displacements and counts to be used for scatterv below (both in terms of # of elements)
    int *displs_x = new int[num_procs];
    int *displs_y = new int[num_procs];
    int *counts_x = new int[num_procs];
    int *counts_y = new int[num_procs];

    for (int i = 0; i < num_procs; i++)
    {
        int last_batch_size = X_n_cols - (num_batches-1)*batch_size;
        int last_minibatch_size = (last_batch_size + num_procs - 1) / num_procs; // new minibatch size for this last batch. last process gets the least.

        displs_x[i] = X_n_rows * last_minibatch_size * i + X_n_rows * (num_batches - 1) * batch_size;
        displs_y[i] = y_n_rows * last_minibatch_size * i + y_n_rows * (num_batches - 1) * batch_size;

        if (last_batch_size % num_procs != 0 && i == num_procs - 1)
        { // if the last process needs to be assigned a few less columns
            counts_x[i] = X_n_rows * (last_batch_size - last_minibatch_size * i);
            counts_y[i] = y_n_rows * (last_batch_size - last_minibatch_size * i);
        }
        else
        {
            counts_x[i] = X_n_rows * last_minibatch_size;
            counts_y[i] = y_n_rows * last_minibatch_size;
        }
    }

    // calc how many elements I expect to receive given my rank 
    int recv_count_x = counts_x[rank];
    int recv_count_y = counts_y[rank];

    // Use scatterv to scatter input data to different processes for the final batch
    arma::mat my_last_X(X_n_rows, recv_count_x / X_n_rows);
    arma::mat my_last_y(y_n_rows, recv_count_y / y_n_rows);
    MPI_Scatterv(X.memptr(), counts_x, displs_x, MPI_DOUBLE, my_last_X.memptr(), recv_count_x, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(y.memptr(), counts_y, displs_y, MPI_DOUBLE, my_last_y.memptr(), recv_count_y, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    my_X_batches.push_back(my_last_X);
    my_y_batches.push_back(my_last_y);


    // ----------------------
    // Batch gradient descent
    // ----------------------
    NeuralNetwork nn_test(nn.H); // delete this and next line eventually 
    double my_tol = 1.0e-6; // no differences for tol greater than 1.0e1. start to get errors at 1.0e0
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (int batch = 0; batch < num_batches; ++batch)
        {
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */
            bool tests = false;

            int my_N = my_X.n_cols;
            int last_col = std::min((batch + 1) * batch_size / num_procs - 1, my_N - 1);
            arma::mat X_batch = my_X.cols(batch * batch_size / num_procs, last_col); // columns at index (batch * batch_size / num_procs) to (last_col) inclusive
            arma::mat y_batch = my_y.cols(batch * batch_size / num_procs, last_col);
            printf("My rank is [%1d] and for this batch I start at column %1d and last_col is %1d.\n", rank, batch * batch_size / num_procs, last_col);
            // int start = batch * batch_size + rank * batch_size/num_procs;
            // last_col = std::min((batch + 1) * batch_size / num_procs - 1, N - 1);;
            // int finish = last_col + start;
            // printf("My rank is [%1d] and for this batch I start at column %1d and end at %1d and last_col is %1d.\n", rank, start, finish, last_col);
            // arma::mat X_batch = X.cols(start, finish);
            // arma::mat y_batch = y.cols(start, finish);
            // exit(EXIT_FAILURE); // reaches here fine

            // std::cout << "rank and displs_x[rank]/X_n_rows is " << rank << " and " << displs_x[rank]/X_n_rows <<std::endl; // to see what part of X each process is starting from

            // checkpoint 1: compare calculated minibatches to what they should be (obtained directly from X)
            if (batch==0 && tests){
                arma::mat test_X = X.cols(displs_x[rank] / X.n_rows, displs_x[rank] / X.n_rows + batch_size / num_procs - 1);
                arma::mat test_y = y.cols(displs_y[rank] / y.n_rows, displs_y[rank] / y.n_rows + batch_size / num_procs - 1);

                // number not equal - no errors here - 3/14/20 
                // std::cout << "Number not equal for X on rank " << rank << " is : " << almost_equal_matrix(test_X, X_batch.memptr(), true) << std::endl;
                // std::cout << "Number not equal for y on rank " << rank << " is : " << almost_equal_matrix(test_y, y_batch.memptr(), true) << std::endl;

                // approx_equal calls 
                if (!arma::approx_equal(X_batch, test_X, "both", my_tol, my_tol))
                {
                    std::cout << "incorrect minibatch for X on rank " << rank << std::endl;
                    // std::cout << X.cols(0, 200).n_rows << " and " << X.cols(0, 200).n_cols << std::endl;
                }
                if (!arma::approx_equal(y_batch, test_y, "both", my_tol, my_tol))
                    std::cout << "incorrect minibatch for y on rank " << rank << std::endl;

                // checkpoint 2: verify nn is same for every process
                if (!arma::approx_equal(nn_test.W[0], nn.W[0], "both", my_tol, my_tol))
                {
                    std::cout << "error with nn.W[0] on rank " << rank << std::endl;
                }
                // else
                // {
                //     std::cout << "norm of nn.W[0] on rank " << rank << " is " << arma::norm(nn.W[0],2) << std::endl;
                // }
                if (!arma::approx_equal(nn_test.W[1], nn.W[1], "both", my_tol, my_tol))
                {
                    std::cout << "error with nn.W[1] on rank " << rank << std::endl;
                }
                // else
                // {
                //     std::cout << "norm of nn.b[1] on rank " << rank << " is " << arma::norm(nn.b[1], 1) << std::endl;
                // }
                if (!arma::approx_equal(nn_test.b[0], nn.b[0], "both", my_tol, my_tol))
                {
                    std::cout << "error with nn.b[0] on rank " << rank << std::endl;
                }
                // else
                // {
                //     std::cout << "norm of nn.b[0] on rank " << rank << " is " << arma::norm(nn.b[0], 1) << std::endl;
                // }
                if (!arma::approx_equal(nn_test.b[1], nn.b[1], "both", my_tol, my_tol))
                {
                    std::cout << "error with nn.b[1] on rank " << rank << std::endl;
                }
                // else
                // {
                //     std::cout << "norm of nn.b[1] on rank " << rank << " is " << arma::norm(nn.b[1], 1) << std::endl;
                // }
                // exit(EXIT_FAILURE); // reaches here fine
            }

            // forward and backward pass
            struct cache bpcache;
            // GPUfeedforward(nn, X_batch, bpcache); // feedforward okay
            feedforward(nn, X_batch, bpcache);
            // exit(EXIT_FAILURE); // reaches here fine

            // checkpoint 3: compare gathering bpcache from each process vs. calculating it entirely on process 0
            if (batch==0 && num_procs==4 && tests){
                arma::mat gather_a0(bpcache.a[0].n_rows, batch_size);
                arma::mat gather_a1(bpcache.a[1].n_rows, batch_size);
                arma::mat gather_z0(bpcache.z[0].n_rows, batch_size);
                arma::mat gather_z1(bpcache.z[1].n_rows, batch_size);
                // exit(EXIT_FAILURE); // reaches here on grading mode 3

                // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                //    void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
                // std::cout << bpcache.a[0].n_rows << ", " << bpcache.a[0].n_cols << std::endl;
                // std::cout << bpcache.z[0].n_rows << ", " << bpcache.z[0].n_cols << std::endl;

                MPI_Gather(bpcache.a[0].memptr(), gather_a0.n_rows * batch_size/num_procs, MPI_DOUBLE, gather_a0.memptr(), gather_a0.n_rows * batch_size/num_procs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gather(bpcache.a[1].memptr(), gather_a1.n_rows * batch_size/num_procs, MPI_DOUBLE, gather_a1.memptr(), gather_a1.n_rows * batch_size/num_procs, MPI_DOUBLE, 0, MPI_COMM_WORLD); // this is okay
                MPI_Gather(bpcache.z[0].memptr(), gather_z0.n_rows * batch_size/num_procs, MPI_DOUBLE, gather_z0.memptr(), gather_z0.n_rows * batch_size/num_procs, MPI_DOUBLE, 0, MPI_COMM_WORLD); // not this one
                MPI_Gather(bpcache.z[1].memptr(), gather_z1.n_rows * batch_size/num_procs, MPI_DOUBLE, gather_z1.memptr(), gather_z1.n_rows * batch_size/num_procs, MPI_DOUBLE, 0, MPI_COMM_WORLD); // this is okay
                // exit(EXIT_FAILURE); // does not reach here on grading mode 3

                if (rank == 0)
                {
                    arma::mat rank0_X = arma::join_rows(X.cols(0, batch_size / num_procs - 1), X.cols(displs_x[1] / X.n_rows, displs_x[1] / X.n_rows + batch_size / num_procs - 1));
                    rank0_X = arma::join_rows(rank0_X, X.cols(displs_x[2] / X.n_rows, displs_x[2] / X.n_rows + batch_size / num_procs - 1));
                    rank0_X = arma::join_rows(rank0_X, X.cols(displs_x[3] / X.n_rows, displs_x[3] / X.n_rows + batch_size / num_procs - 1));

                    struct cache bpcache_test;
                    feedforward(nn_test, rank0_X, bpcache_test);

                    // for testing one minibatch at a time. rank 0 is good for tol = 1.0e-12.
                    // if (!arma::approx_equal(bpcache_test.a[0], bpcache.a[0], "both", my_tol, my_tol))
                    // {
                    //     std::cout << "Problem with a[0] for rank" << rank << " minibatch" << std::endl;
                    //     std::cout << arma::norm(bpcache_test.a[0], 1) << " vs " << arma::norm(bpcache.a[0], 1) << std::endl;
                    // }
                    // if (!arma::approx_equal(bpcache_test.a[1], bpcache.a[1], "both", my_tol, my_tol))
                    // {
                    //     std::cout << "Problem with a[0] for rank" << rank << " minibatch" << std::endl;
                    //     std::cout << arma::norm(bpcache_test.a[1], 1) << " vs " << arma::norm(bpcache.a[1], 1) << std::endl;
                    // }
                    // if (!arma::approx_equal(bpcache_test.z[0], bpcache.z[0], "both", my_tol, my_tol))
                    // {
                    //     std::cout << "Problem with a[0] for rank" << rank << " minibatch" << std::endl;
                    //     std::cout << arma::norm(bpcache_test.z[0], 1) << " vs " << arma::norm(bpcache.z[0], 1) << std::endl;
                    // }
                    // if (!arma::approx_equal(bpcache_test.z[1], bpcache.z[1], "both", my_tol, my_tol))
                    // {
                    //     std::cout << "Problem with a[0] for for rank" << rank << " minibatch" << std::endl;
                    //     std::cout << arma::norm(bpcache_test.z[1], 1) << " vs " << arma::norm(bpcache.z[1], 1) << std::endl;
                    // }

                    // count number wrong
                    // std::cout << "Number not equal for a[0] is: " << almost_equal_matrix(bpcache_test.a[0], gather_a0.memptr(), true) << std::endl;
                    // std::cout << "Number not equal for a[1] is: " << almost_equal_matrix(bpcache_test.a[1], gather_a1.memptr(), true) << std::endl;
                    // std::cout << "Number not equal for z[0] is: " << almost_equal_matrix(bpcache_test.z[0], gather_z0.memptr(), true) << std::endl;
                    // std::cout << "Number not equal for z[1] is: " << almost_equal_matrix(bpcache_test.z[1], gather_z1.memptr(), true) << std::endl;

                    // approx_equal tests 
                    assert(gather_a0.n_rows == bpcache_test.a[0].n_rows);
                    assert(gather_a0.n_cols == bpcache_test.a[0].n_cols);
                    assert(gather_z1.n_rows == bpcache_test.z[1].n_rows);
                    assert(gather_z1.n_cols == bpcache_test.z[1].n_cols);

                    if (!arma::approx_equal(bpcache_test.a[0], gather_a0, "both", my_tol, my_tol))
                    {
                        std::cout << "Problem with a[0]" << std::endl;
                        std::cout << arma::norm(bpcache_test.a[0], 1) << " vs " << arma::norm(gather_a0, 1) << std::endl;
                    }
                    if (!arma::approx_equal(bpcache_test.a[1], gather_a1, "both", my_tol, my_tol))
                    {
                        std::cout << "Problem with a[1]" << std::endl;
                        std::cout << arma::norm(bpcache_test.a[1], 1) << " vs " << arma::norm(gather_a1, 1) << std::endl;
                    }
                    if (!arma::approx_equal(bpcache_test.z[0], gather_z0, "both", my_tol, my_tol))
                    {
                        printf("\n");
                        std::cout << "Problem with z[0]" << std::endl;
                        std::cout << arma::norm(bpcache_test.z[0], 1) << " vs " << arma::norm(gather_z0, 1) << std::endl;

                        arma::mat z_diff = abs(bpcache_test.z[0] - gather_z0);
                        std::cout << "max difference is: " << z_diff.max() << " at linear index: " << z_diff.index_max() << std::endl;
                        std::cout << "At that location bpcache_test.z[0] and gather_z0 are: " << bpcache_test.z[0][z_diff.index_max()] << " and " << gather_z0[z_diff.index_max()] << std::endl;
                        std::cout << "min difference is: " << z_diff.min() << " at linear index: " << z_diff.index_min() << std::endl;
                        std::cout << "bpcache.z[0] has dimensions: " << bpcache_test.z[0].n_rows << " x " << bpcache_test.z[0].n_cols << std::endl;
                        // index 36906 is somewhere in rank 1's block
                        printf("\n");
                    }
                    if (!arma::approx_equal(bpcache_test.z[1], gather_z1, "both", my_tol, my_tol))
                    {
                        std::cout << "Problem with z[1]" << std::endl;
                        std::cout << arma::norm(bpcache_test.z[1], 1) << " vs " << arma::norm(gather_z1, 1) << std::endl;
                    }
                    // exit(EXIT_FAILURE); // reaches here fine
                }
                // well, feedforward definitely works
            }

            struct grads bpgrads;
            // GPUbackprop(nn, y_batch, reg, bpcache, bpgrads, num_procs); // backprop okay
            // std::cout << "my rank and num_procs is: " << rank << " and " << num_procs << std::endl; 
            backprop2(nn, y_batch, reg, bpcache, bpgrads, num_procs); // reg/4 here 

            // checkpoint 4: check local gradients
            if (batch==0 && tests){
                // std::cout << "rank and displs_x[rank]/X_n_rows is " << rank << " and " << displs_x[rank] / X_n_rows << std::endl; // to see what part of X each process is starting from
                NeuralNetwork nn_test4(nn.H);

                arma::mat minibatch_X = X.cols(displs_x[rank] / X_n_rows, displs_x[rank] / X_n_rows + batch_size/num_procs - 1);
                arma::mat minibatch_y = y.cols(displs_y[rank] / y_n_rows, displs_y[rank] / y_n_rows + batch_size/num_procs - 1);

                struct cache bpcache_test;
                feedforward(nn_test4, minibatch_X, bpcache_test);

                struct grads bpgrads_test;
                backprop2(nn_test4, minibatch_y, reg, bpcache_test, bpgrads_test, num_procs);

                // count number wrong - 3/14/20 all of these are reporting no errors!
                // std::cout << "Number not equal for dW[0] on rank " << rank << " is: " << almost_equal_matrix(bpgrads.dW[0], bpgrads_test.dW[0].memptr(), true) << std::endl; // good 
                // std::cout << "Number not equal for dW[1] on rank " << rank << " is: " << almost_equal_matrix(bpgrads.dW[1], bpgrads_test.dW[1].memptr(), true) << std::endl; // good 
                // std::cout << "Number not equal for db[0] on rank " << rank << " is: " << almost_equal_matrix(bpgrads.db[0], bpgrads_test.db[0].memptr(), true) << std::endl;
                // std::cout << "Number not equal for db[1] on rank " << rank << " is: " << almost_equal_matrix(bpgrads.db[1], bpgrads_test.db[1].memptr(), true) << std::endl;
                // std::cout << "Number not equal for dz1 on rank " << rank << " is: " << almost_equal_matrix(bpgrads.dz1, bpgrads_test.dz1.memptr(), true) << std::endl; 
                // std::cout << "Number not equal for X.t on rank " << rank << " is: " << almost_equal_matrix(bpgrads.Xt, bpgrads_test.Xt.memptr(), true) << std::endl;
                // std::cout << "Number not equal for w0 on rank " << rank << " is: " << almost_equal_matrix(bpgrads.w0, bpgrads_test.w0.memptr(), true) << std::endl;
                // std::cout << "Number not equal for da1 on rank " << rank << " is: " << almost_equal_matrix(bpgrads.da1, bpgrads_test.da1.memptr(), true) << std::endl; 
                // std::cout << "Number not equal for a0 on rank " << rank << " is: " << almost_equal_matrix(bpgrads.a0, bpgrads_test.a0.memptr(), true) << std::endl; // good
                // std::cout << "Number not equal for diff on rank " << rank << " is: " << almost_equal_matrix(bpgrads.diff, bpgrads_test.diff.memptr(), true) << std::endl; // good 
                // std::cout << "Number not equal for w1 on rank " << rank << " is: " << almost_equal_matrix(bpgrads.w1, bpgrads_test.w1.memptr(), true) << std::endl; // good 
                // std::cout << "Number not equal for a0t on rank " << rank << " is: " << almost_equal_matrix(bpgrads.a0t, bpgrads_test.a0t.memptr(), true) << std::endl; // good 



                // approx_equal tests
                if (!arma::approx_equal(bpgrads_test.dW[0], bpgrads.dW[0], "both", my_tol, my_tol))
                {
                    std::cout << "Problem with dW[0] for rank " << rank << " minibatch" << std::endl;
                    std::cout << arma::norm(bpgrads_test.dW[0], 1) << " vs " << arma::norm(bpgrads.dW[0], 1) << std::endl;
                }
                if (!arma::approx_equal(bpgrads_test.dW[1], bpgrads.dW[1], "both", my_tol, my_tol))
                {
                    std::cout << "Problem with dW[1] for rank " << rank << " minibatch" << std::endl;
                    std::cout << arma::norm(bpgrads_test.dW[1], 1) << " vs " << arma::norm(bpgrads.dW[1], 1) << std::endl;
                }
                if (!arma::approx_equal(bpgrads_test.db[0], bpgrads.db[0], "both", my_tol, my_tol))
                {
                    std::cout << "Problem with db[0] for rank " << rank << " minibatch" << std::endl;
                    std::cout << arma::norm(bpgrads_test.db[0], 1) << " vs " << arma::norm(bpgrads.db[0], 1) << std::endl;
                }
                if (!arma::approx_equal(bpgrads_test.db[1], bpgrads.db[1], "both", my_tol, my_tol))
                {
                    std::cout << "Problem with db[1] for for rank " << rank << " minibatch" << std::endl;
                    std::cout << arma::norm(bpgrads_test.db[1], 1) << " vs " << arma::norm(bpgrads.db[1], 1) << std::endl;
                }
            }

            // // Allocate memory for global gradients
            // double *dW0 = new double[bpgrads.dW[0].n_rows * bpgrads.dW[0].n_cols];
            // double *dW1 = new double[bpgrads.dW[1].n_rows * bpgrads.dW[1].n_cols];
            // double *db0 = new double[bpgrads.db[0].n_rows * bpgrads.db[0].n_cols];
            // double *db1 = new double[bpgrads.db[1].n_rows * bpgrads.db[1].n_cols];

            // // MPI all reduce local bpgrads
            // MPI_Allreduce(bpgrads.dW[0].memptr(), dW0, bpgrads.dW[0].n_rows * bpgrads.dW[0].n_cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // MPI_Allreduce(bpgrads.dW[1].memptr(), dW1, bpgrads.dW[1].n_rows * bpgrads.dW[1].n_cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // MPI_Allreduce(bpgrads.db[0].memptr(), db0, bpgrads.db[0].n_rows * bpgrads.db[0].n_cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // MPI_Allreduce(bpgrads.db[1].memptr(), db1, bpgrads.db[1].n_rows * bpgrads.db[1].n_cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // // std::cout << "hello from rank " << rank << " where I allocated " << bpgrads.db[1].n_rows * bpgrads.db[1].n_cols << " for db1 " << std::endl; // all are same

            // // transfer reduced bpgrads to each gpu
            // cudaMemcpy(d_dW0, dW0, sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyHostToDevice);
            // cudaMemcpy(d_dW1, dW1, sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyHostToDevice);
            // cudaMemcpy(d_db0, db0, sizeof(double) * nn.b[0].n_rows * nn.b[0].n_cols, cudaMemcpyHostToDevice);
            // cudaMemcpy(d_db1, db1, sizeof(double) * nn.b[1].n_rows * nn.b[1].n_cols, cudaMemcpyHostToDevice);
            // // std::cout << "On rank " << rank << " I have local and global gradient dW0[100]: " << bpgrads.dW[0][100] << ", " << dW0[100] << std::endl; // this makes it look like all procces are calculating the same local gradients?

            // // Gradient descent - W0
            // // std::cout << "hello from rank " << rank << " where I nn.W[0].n_rows is:  " << nn.W[0].n_rows << std::endl; // all are same
            // GPUaddition(d_W0, d_dW0, d_W0, 1.0, -learning_rate, nn.W[0].n_rows, nn.W[0].n_cols);
            // arma::mat W0(nn.W[0].n_rows, nn.W[0].n_cols);
            // cudaMemcpy(W0.memptr(), d_W0, sizeof(double) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyDeviceToHost);
            // nn.W[0] = W0;
            // // std::cout << "rank and norm(nn.W[0]): " << rank << " and " << norm(nn.W[0], 1) << std::endl; //

            // // Gradient descent - W1
            // GPUaddition(d_W1, d_dW1, d_W1, 1.0, -learning_rate, nn.W[1].n_rows, nn.W[1].n_cols);
            // arma::mat W1(nn.W[1].n_rows, nn.W[1].n_cols);
            // cudaMemcpy(W1.memptr(), d_W1, sizeof(double) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyDeviceToHost);
            // nn.W[1] = W1;
            // // std::cout << "rank and norm(nn.W[1]): " << rank << " and " << norm(nn.W[1], 1) << std::endl; //

            // // Gradient descent - b0
            // GPUaddition(d_b0, d_db0, d_b0, 1.0, -learning_rate, nn.b[0].n_rows, nn.b[0].n_cols);
            // arma::mat b0(nn.b[0].n_rows, nn.b[0].n_cols);
            // cudaMemcpy(b0.memptr(), d_b0, sizeof(double) * nn.b[0].n_rows * nn.b[0].n_cols, cudaMemcpyDeviceToHost);
            // nn.b[0] = b0;
            // // std::cout << "rank and norm(nn.b[0]): " << rank << " and " << norm(nn.b[0], 1) << std::endl; // each b value is exactly double what it should be when you use -n 2

            // // Gradient descent - b1
            // GPUaddition(d_b1, d_db1, d_b1, 1.0, -learning_rate, nn.b[1].n_rows, nn.b[1].n_cols);
            // arma::mat b1(nn.b[1].n_rows, nn.b[1].n_cols);
            // cudaMemcpy(b1.memptr(), d_b1, sizeof(double) * nn.b[1].n_rows * nn.b[1].n_cols, cudaMemcpyDeviceToHost);
            // nn.b[1] = b1;
            // // std::cout << "rank and norm(nn.b[1]): " << rank << " and " << norm(nn.b[1], 1) << std::endl; //
            // // exit(EXIT_FAILURE);

            // checkpoint 5: compare network coefficients before gradient descent
            if (batch==0 && tests){
                NeuralNetwork nn_test5(nn.H);
                if (!arma::approx_equal(nn_test5.W[0], nn.W[0], "both", my_tol, my_tol))
                {
                    std::cout << "Problem with W[0] before GD" << std::endl;
                    std::cout << arma::norm(nn_test5.W[1], 1) << " vs " << arma::norm(nn.W[0], 1) << std::endl;
                }
                if (!arma::approx_equal(nn_test5.W[1], nn.W[1], "both", my_tol, my_tol))
                {
                    std::cout << "Problem with W[1] before GD" << std::endl;
                    std::cout << arma::norm(nn_test5.W[1], 1) << " vs " << arma::norm(nn.W[1], 1) << std::endl;
                }
                if (!arma::approx_equal(nn_test5.b[0], nn.b[0], "both", my_tol, my_tol))
                {
                    std::cout << "Problem with b[0] before GD" << std::endl;
                    std::cout << arma::norm(nn_test5.b[0], 1) << " vs " << arma::norm(nn.b[0], 1) << std::endl;
                }
                if (!arma::approx_equal(nn_test5.b[1], nn.b[1], "both", my_tol, my_tol))
                {
                    std::cout << "Problem with b[1] before GD" << std::endl;
                    std::cout << arma::norm(nn_test5.b[1], 1) << " vs " << arma::norm(nn.b[1], 1) << std::endl;
                }
            }

            // Gradient descent step on CPU. gives same result as gpu calculations above.
            // arma::mat W0(nn.W[0].n_rows, nn.W[0].n_cols);
            // arma::mat W1(nn.W[1].n_rows, nn.W[1].n_cols);
            // arma::mat b0(nn.b[0].n_rows, nn.b[0].n_cols);
            // arma::mat b1(nn.b[1].n_rows, nn.b[1].n_cols);
            arma::mat W0 = arma::zeros(nn.W[0].n_rows, nn.W[0].n_cols);
            arma::mat W1 = arma::zeros(nn.W[1].n_rows, nn.W[1].n_cols);
            arma::mat b0 = arma::zeros(nn.b[0].n_rows, nn.b[0].n_cols);
            arma::mat b1 = arma::zeros(nn.b[1].n_rows, nn.b[1].n_cols);
            // double* W0 = new double[nn.W[0].n_elem];
            // double* W1 = new double[nn.W[1].n_elem];
            // double* b0 = new double[nn.b[0].n_elem];
            // double* b1 = new double[nn.b[1].n_elem];
            // memset(W0,0,8*nn.W[0].n_elem);
            // memset(W1,0,8*nn.W[0].n_elem);
            // memset(b0,0,8*nn.W[0].n_elem);
            // memset(b1,0,8*nn.W[0].n_elem);
            // MPI_Allreduce(bpgrads.dW[0].memptr(), W0, nn.W[0].n_elem, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // MPI_Allreduce(bpgrads.dW[1].memptr(), W1, nn.W[1].n_elem, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // MPI_Allreduce(bpgrads.db[0].memptr(), b0, nn.b[0].n_elem, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // MPI_Allreduce(bpgrads.db[1].memptr(), b1, nn.b[1].n_elem, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(bpgrads.dW[0].memptr(), W0.memptr(), nn.W[0].n_elem, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(bpgrads.dW[1].memptr(), W1.memptr(), nn.W[1].n_elem, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(bpgrads.db[0].memptr(), b0.memptr(), nn.b[0].n_elem, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(bpgrads.db[1].memptr(), b1.memptr(), nn.b[1].n_elem, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            nn.W[0] -= learning_rate * W0;
            nn.W[1] -= learning_rate * W1;
            nn.b[0] -= learning_rate * b0;
            nn.b[1] -= learning_rate * b1;

            // checkpoint 6: instead of allreduce, sum each local gradient manually 
            if (batch == 0 && num_procs==4 && tests){
                NeuralNetwork nn_test(nn.H);

                // create input
                std::vector<arma::mat> batch_X;
                batch_X.push_back(X.cols(0, batch_size / num_procs - 1));
                batch_X.push_back(X.cols(displs_x[1] / X_n_rows, displs_x[1] / X_n_rows + batch_size / num_procs - 1));
                batch_X.push_back(X.cols(displs_x[2] / X_n_rows, displs_x[2] / X_n_rows + batch_size / num_procs - 1));
                batch_X.push_back(X.cols(displs_x[3] / X_n_rows, displs_x[3] / X_n_rows + batch_size / num_procs - 1));

                std::vector<arma::mat> batch_y;
                batch_y.push_back(y.cols(0, batch_size / num_procs - 1));
                batch_y.push_back(y.cols(displs_y[1] / y_n_rows, displs_y[1] / y_n_rows + batch_size / num_procs - 1));
                batch_y.push_back(y.cols(displs_y[2] / y_n_rows, displs_y[2] / y_n_rows + batch_size / num_procs - 1));
                batch_y.push_back(y.cols(displs_y[3] / y_n_rows, displs_y[3] / y_n_rows + batch_size / num_procs - 1));

                // checks out - 3/14/20 
                // std::cout << "Number not equal for X on rank " << rank << " is : " << almost_equal_matrix(batch_X[rank], X_batch.memptr(), true) << std::endl;
                // std::cout << "Number not equal for y on rank " << rank << " is : " << almost_equal_matrix(batch_y[rank], y_batch.memptr(), true) << std::endl;

                // feedforward with each cache ... these all check out 
                struct cache bpcache1;
                struct cache bpcache2;
                struct cache bpcache3;
                struct cache bpcache4;
                feedforward(nn_test, batch_X[0], bpcache1);
                feedforward(nn_test, batch_X[1], bpcache2);
                feedforward(nn_test, batch_X[2], bpcache3);
                feedforward(nn_test, batch_X[3], bpcache4);
                std::vector<cache> bpcache_test;
                bpcache_test.push_back(bpcache1);
                bpcache_test.push_back(bpcache2);
                bpcache_test.push_back(bpcache3);
                bpcache_test.push_back(bpcache4);
                // std::cout << "Number not equal for a[0] is: " << almost_equal_matrix(bpcache.a[0], bpcache_test[rank].a[0].memptr(), true) << std::endl;
                // std::cout << "Number not equal for a[1] is: " << almost_equal_matrix(bpcache.a[1], bpcache_test[rank].a[1].memptr(), true) << std::endl;
                // std::cout << "Number not equal for z[0] is: " << almost_equal_matrix(bpcache.z[0], bpcache_test[rank].z[0].memptr(), true) << std::endl;
                // std::cout << "Number not equal for z[1] is: " << almost_equal_matrix(bpcache.z[1], bpcache_test[rank].z[1].memptr(), true) << std::endl;

                // backprop with each cache ... these all check out again!
                struct grads bpgrads1;
                struct grads bpgrads2;
                struct grads bpgrads3;
                struct grads bpgrads4;
                backprop2(nn_test, batch_y[0], reg, bpcache1, bpgrads1, num_procs);
                backprop2(nn_test, batch_y[1], reg, bpcache2, bpgrads2, num_procs);
                backprop2(nn_test, batch_y[2], reg, bpcache3, bpgrads3, num_procs);
                backprop2(nn_test, batch_y[3], reg, bpcache4, bpgrads4, num_procs);
                std::vector<grads> bpgrads_test;
                bpgrads_test.push_back(bpgrads1);
                bpgrads_test.push_back(bpgrads2);
                bpgrads_test.push_back(bpgrads3);
                bpgrads_test.push_back(bpgrads4);
                // std::cout << "Number not equal for dW[0] is: " << almost_equal_matrix(bpgrads_test[rank].dW[0], bpgrads.dW[0].memptr(), true) << std::endl;
                // std::cout << "Number not equal for dW[1] is: " << almost_equal_matrix(bpgrads_test[rank].dW[1], bpgrads.dW[1].memptr(), true) << std::endl;
                // std::cout << "Number not equal for db[0] is: " << almost_equal_matrix(bpgrads_test[rank].db[0], bpgrads.db[0].memptr(), true) << std::endl;
                // std::cout << "Number not equal for db[1] is: " << almost_equal_matrix(bpgrads_test[rank].db[1], bpgrads.db[1].memptr(), true) << std::endl;

                // now check total gradients
                arma::mat total_dW0 = arma::zeros(bpgrads.dW[0].n_rows, bpgrads.dW[0].n_cols);
                arma::mat total_dW1 = arma::zeros(bpgrads.dW[1].n_rows, bpgrads.dW[1].n_cols);
                arma::mat total_db0 = arma::zeros(bpgrads.db[0].n_rows, bpgrads.db[0].n_cols);
                arma::mat total_db1 = arma::zeros(bpgrads.db[1].n_rows, bpgrads.db[1].n_cols);
                for (int i = 0; i < num_procs; i++){
                    total_dW0 = total_dW0 + bpgrads_test[i].dW[0];
                    total_dW1 = total_dW1 + bpgrads_test[i].dW[1];
                    total_db0 = total_db0 + bpgrads_test[i].db[0];
                    total_db1 = total_db1 + bpgrads_test[i].db[1];
                }

                // only very few errors in dW[0] here 
                // std::cout << "Number not equal for dW[0] is: " << almost_equal_matrix(total_dW0, W0.memptr(), true) << std::endl; // 3 here
                // std::cout << "Number not equal for dW[1] is: " << almost_equal_matrix(total_dW1, W1.memptr(), true) << std::endl; // 0
                // std::cout << "Number not equal for db[0] is: " << almost_equal_matrix(total_db0, b0.memptr(), true) << std::endl; // 0 
                // std::cout << "Number not equal for db[1] is: " << almost_equal_matrix(total_db1, b1.memptr(), true) << std::endl; // 0

                // update network coefficients - these are all good though!
                nn_test.W[0] -= learning_rate * total_dW0;
                nn_test.W[1] -= learning_rate * total_dW1;
                nn_test.b[0] -= learning_rate * total_db0;
                nn_test.b[1] -= learning_rate * total_db1;
                std::cout << "Number not equal for W[0] is: " << almost_equal_matrix(nn.W[0], nn_test.W[0].memptr(), true) << std::endl; // again 3 here
                std::cout << "Number not equal for W[1] is: " << almost_equal_matrix(nn.W[1], nn_test.W[1].memptr(), true) << std::endl;
                std::cout << "Number not equal for b[0] is: " << almost_equal_matrix(nn.b[0], nn_test.b[0].memptr(), true) << std::endl;
                std::cout << "Number not equal for b[1] is: " << almost_equal_matrix(nn.b[1], nn_test.b[1].memptr(), true) << std::endl;

                // checkpoint 7: check total gradients
                NeuralNetwork nn_test2(nn.H);

                arma::mat batch_X2 = arma::join_rows(X.cols(0, batch_size / num_procs - 1), X.cols(displs_x[1] / X_n_rows, displs_x[1] / X_n_rows + batch_size / num_procs - 1));
                batch_X2 = arma::join_rows(batch_X2, X.cols(displs_x[2] / X_n_rows, displs_x[2] / X_n_rows + batch_size / num_procs - 1));
                batch_X2 = arma::join_rows(batch_X2, X.cols(displs_x[3] / X_n_rows, displs_x[3] / X_n_rows + batch_size / num_procs - 1));

                arma::mat batch_y2 = arma::join_rows(y.cols(0, batch_size / num_procs - 1), y.cols(displs_y[1] / y_n_rows, displs_y[1] / y_n_rows + batch_size / num_procs - 1));
                batch_y2 = arma::join_rows(batch_y2, y.cols(displs_y[2] / y_n_rows, displs_y[2] / y_n_rows + batch_size / num_procs - 1));
                batch_y2 = arma::join_rows(batch_y2, y.cols(displs_y[3] / y_n_rows, displs_y[3] / y_n_rows + batch_size / num_procs - 1));

                // concatenation is not the problem 
                arma::mat combined_X = arma::join_rows(batch_X[0], batch_X[1]);
                combined_X = arma::join_rows(combined_X, batch_X[2]);
                combined_X = arma::join_rows(combined_X, batch_X[3]);
                arma::mat combined_y = arma::join_rows(batch_y[0], batch_y[1]);
                combined_y = arma::join_rows(combined_y, batch_y[2]);
                combined_y = arma::join_rows(combined_y, batch_y[3]);
                // std::cout << "Number not equal for X on rank " << rank << " is : " << almost_equal_matrix(combined_X, batch_X2.memptr(), true) << std::endl;
                // std::cout << "Number not equal for y on rank " << rank << " is : " << almost_equal_matrix(combined_y, batch_y2.memptr(), true) << std::endl;
                // std::cout << arma::norm(combined_X, 2) << " vs " << arma::norm(batch_X2, 2) << std::endl;
                // std::cout << arma::norm(combined_y, 2) << " vs " << arma::norm(batch_y2, 2) << std::endl;

                // feedforward and backprop
                struct cache bpcache_test2;
                feedforward(nn_test2, batch_X2, bpcache_test2);

                // all good here 
                // std::cout << "Number not equal for a[0] is: " << almost_equal_matrix(bpcache_test2.a[0].cols(rank * batch_size / num_procs, (rank + 1) * batch_size / num_procs - 1), bpcache_test[rank].a[0].memptr(), true) << std::endl;
                // std::cout << "Number not equal for a[1] is: " << almost_equal_matrix(bpcache_test2.a[1].cols(rank * batch_size / num_procs, (rank + 1) * batch_size / num_procs - 1), bpcache_test[rank].a[1].memptr(), true) << std::endl;
                // std::cout << "Number not equal for z[0] is: " << almost_equal_matrix(bpcache_test2.z[0].cols(rank * batch_size / num_procs, (rank + 1) * batch_size / num_procs - 1), bpcache_test[rank].z[0].memptr(), true) << std::endl;
                // std::cout << "Number not equal for z[1] is: " << almost_equal_matrix(bpcache_test2.z[1].cols(rank * batch_size / num_procs, (rank + 1) * batch_size / num_procs - 1), bpcache_test[rank].z[1].memptr(), true) << std::endl;
                // exit(EXIT_FAILURE);

                struct grads bpgrads_test2;
                backprop(nn_test2, batch_y2, reg, bpcache_test2, bpgrads_test2); 

                assert(bpgrads_test2.dW[0].n_rows==W0.n_rows);
                assert(bpgrads_test2.dW[0].n_cols == W0.n_cols);
                assert(bpgrads_test2.dW[1].n_rows == W1.n_rows);
                assert(bpgrads_test2.dW[1].n_cols == W1.n_cols);

                // this is where we start to see errors
                // std::cout << "Number not equal for dW[0] is: " << almost_equal_matrix(bpgrads_test2.dW[0], W0.memptr(), true) << std::endl;
                // std::cout << "Number not equal for dW[1] is: " << almost_equal_matrix(bpgrads_test2.dW[1], W1.memptr(), true) << std::endl;
                // std::cout << "Number not equal for db[0] is: " << almost_equal_matrix(bpgrads_test2.db[0], b0.memptr(), true) << std::endl;
                // std::cout << "Number not equal for db[1] is: " << almost_equal_matrix(bpgrads_test2.db[1], b1.memptr(), true) << std::endl;

                // so it looks like I truly get a different answer depending on if I start with the input combined or not 
                // std::cout << "Number not equal for dW[0] (test2) is: " << almost_equal_matrix(bpgrads_test2.dW[0], total_dW0.memptr(), true) << std::endl;
                // std::cout << "Number not equal for dW[1] (test2) is: " << almost_equal_matrix(bpgrads_test2.dW[1], total_dW1.memptr(), true) << std::endl;
                // std::cout << "Number not equal for db[0] (test2) is: " << almost_equal_matrix(bpgrads_test2.db[0], total_db0.memptr(), true) << std::endl;
                // std::cout << "Number not equal for db[1] (test2) is: " << almost_equal_matrix(bpgrads_test2.db[1], total_db1.memptr(), true) << std::endl;

                // Gradient descent step
                for (int i = 0; i < nn.W.size(); ++i)
                {
                    nn_test2.W[i] -= learning_rate * bpgrads_test2.dW[i];
                }

                for (int i = 0; i < nn.b.size(); ++i)
                {
                    nn_test2.b[i] -= learning_rate * bpgrads_test2.db[i];
                }
                std::cout << "Number not equal for W[0] (test2) is: " << almost_equal_matrix(nn.W[0], nn_test2.W[0].memptr(), true) << std::endl; // again 3 here
                std::cout << "Number not equal for W[1] (test2) is: " << almost_equal_matrix(nn.W[1], nn_test2.W[1].memptr(), true) << std::endl;
                std::cout << "Number not equal for b[0] (test2) is: " << almost_equal_matrix(nn.b[0], nn_test2.b[0].memptr(), true) << std::endl;
                std::cout << "Number not equal for b[1] (test2) is: " << almost_equal_matrix(nn.b[1], nn_test2.b[1].memptr(), true) << std::endl;
                std::cout << "Max difference for W[0] is: " << (nn.W[0]-nn_test2.W[0]).max() << std::endl;
                std::cout << "Max difference for W[1] is: " << (nn.W[1] - nn_test2.W[1]).max() << std::endl;

                // approx_equal tests
                // if (!arma::approx_equal(bpgrads_test2.dW[0], W0, "both", my_tol, my_tol))
                // {
                //     printf("\n");
                //     std::cout << "Problem with dW[0]" << std::endl;
                //     std::cout << "1 norm of dW[0] is: " << arma::norm(bpgrads_test2.dW[0], 1) << " vs " << arma::norm(W0, 1) << std::endl;

                //     arma::mat diff = abs(bpgrads_test2.dW[0] - W0);
                //     std::cout << "W0 has " << W0.n_rows << " and " << W0.n_cols << " columns" << std::endl; 
                //     std::cout << "max difference is: " << diff.max() << " at linear index: " << diff.index_max() << std::endl;
                //     std::cout << "this corresponds to row " << diff.index_max()%W0.n_rows << " and column " << (diff.index_max()-diff.index_max()%W0.n_rows)/W0.n_rows << std::endl; 
                //     std::cout << "At that location dW_test[0] and dW[0] are: " << bpgrads_test2.dW[0][diff.index_max()] << " and " << W0[diff.index_max()] << std::endl;
                //     std::cout << "Number not equal is: " << almost_equal_matrix(bpgrads_test2.dW[0], W0.memptr(), true) << std::endl; 
                //     // exit(EXIT_FAILURE);
                // }
                // if (!arma::approx_equal(bpgrads_test2.dW[1], W1, "both", my_tol, my_tol))
                // {
                //     printf("\n");
                //     std::cout << "Problem with dW[1]" << std::endl;
                //     std::cout << "1 norm of dW[1] is: " << arma::norm(bpgrads_test2.dW[1], 1) << " vs " << arma::norm(W1, 1) << std::endl;

                //     arma::mat diff = abs(bpgrads_test2.dW[1] - W1);
                //     std::cout << "W1 has " << W1.n_rows << " and " << W1.n_cols << " columns" << std::endl;
                //     std::cout << "max difference is: " << diff.max() << " at linear index: " << diff.index_max() << std::endl;
                //     std::cout << "this corresponds to row " << diff.index_max() % W1.n_rows << " and column " << (diff.index_max() - diff.index_max() % W1.n_rows) / W1.n_rows << std::endl;
                //     std::cout << "At that location dW_test[1] and dW[1] are: " << bpgrads_test2.dW[1][diff.index_max()] << " and " << W1[diff.index_max()] << std::endl;
                //     std::cout << "Number not equal is: " << almost_equal_matrix(bpgrads_test2.dW[1], W1.memptr(), true) << std::endl;
                //     printf("\n");
                //     // exit(EXIT_FAILURE);
                // }
                // if (!arma::approx_equal(bpgrads_test2.db[0], b0, "both", my_tol, my_tol))
                // {
                //     std::cout << "Problem with db[0]" << std::endl;
                //     std::cout << arma::norm(bpgrads_test2.db[0], 1) << " vs " << arma::norm(b0, 1) << std::endl;
                // }
                // if (!arma::approx_equal(bpgrads_test2.db[1], b1, "both", my_tol, my_tol))
                // {
                //     std::cout << "Problem with db[1]" << std::endl;
                //     std::cout << arma::norm(bpgrads_test2.db[1], 1) << " vs " << arma::norm(b1, 1) << std::endl;
                // }
                // exit(EXIT_FAILURE);
            }


            // compare to gold standard acting on this batch
            if (rank == 0 && batch==0 && num_procs==4 && tests)
            {
                NeuralNetwork nn_test5(nn.H);

                // create X input
                arma::mat rank0_X = arma::join_rows(X.cols(0, batch_size / num_procs - 1), X.cols(displs_x[1] / X.n_rows, displs_x[1] / X.n_rows + batch_size / num_procs - 1));
                rank0_X = arma::join_rows(rank0_X, X.cols(displs_x[2] / X.n_rows, displs_x[2] / X.n_rows + batch_size / num_procs - 1));
                rank0_X = arma::join_rows(rank0_X, X.cols(displs_x[3] / X.n_rows, displs_x[3] / X.n_rows + batch_size / num_procs - 1));
                // std::cout << X.n_cols << " and " << rank0_X.n_cols << std::endl;

                // create y input
                arma::mat rank0_y = arma::join_rows(y.cols(0, batch_size / num_procs - 1), y.cols(displs_y[1] / y.n_rows, displs_y[1] / y.n_rows + batch_size / num_procs - 1));
                rank0_y = arma::join_rows(rank0_y, y.cols(displs_y[2] / y.n_rows, displs_y[2] / y.n_rows + batch_size / num_procs - 1));
                rank0_y = arma::join_rows(rank0_y, y.cols(displs_y[3] / y.n_rows, displs_y[3] / y.n_rows + batch_size / num_procs - 1));
                // std::cout << y.n_cols << " and " << rank0_y.n_cols << std::endl;

                struct cache bpcache_test;
                feedforward(nn_test5, rank0_X, bpcache_test);

                struct grads bpgrads_test;
                backprop(nn_test5, rank0_y, reg, bpcache_test, bpgrads_test);

                // Gradient descent step
                for (int i = 0; i < nn.W.size(); ++i)
                {
                    nn_test5.W[i] -= learning_rate * bpgrads_test.dW[i];
                }

                for (int i = 0; i < nn.b.size(); ++i)
                {
                    nn_test5.b[i] -= learning_rate * bpgrads_test.db[i];
                }

                std::cout << "Number not equal for W[0] is: " << almost_equal_matrix(nn_test5.W[0], nn.W[0].memptr(), true) << std::endl;
                std::cout << "Number not equal for W[1] is: " << almost_equal_matrix(nn_test5.W[1], nn.W[1].memptr(), true) << std::endl;
                std::cout << "Number not equal for b[0] is: " << almost_equal_matrix(nn_test5.b[0], nn.b[0].memptr(), true) << std::endl;
                std::cout << "Number not equal for b[1] is: " << almost_equal_matrix(nn_test5.b[1], nn.b[1].memptr(), true) << std::endl;

                // checkpoint 7: compare final network coefficients computed distributed with MPI or all on rank 0
                if (!arma::approx_equal(nn_test5.W[0], nn.W[0], "both", my_tol, my_tol))
                {
                    printf("\n");
                    std::cout << "Problem with W[0] for batch 0" << std::endl;
                    std::cout << "norm is " << arma::norm(nn_test5.W[0], 1) << " vs " << arma::norm(nn.W[0], 1) << std::endl;

                    arma::mat diff = abs(nn_test5.W[0] - nn.W[0]);
                    std::cout << "max difference is: " << diff.max() << " at linear index: " << diff.index_max() << std::endl;
                    std::cout << "At that location W_test[0] and W[0] are: " << nn_test5.W[0][diff.index_max()] << " and " << nn_test5.W[0][diff.index_max()] << std::endl;
                    // std::cout << "min difference is: " << diff.min() << " at linear index: " << diff.index_min() << std::endl;
                    // std::cout << "W[0] has dimensions: " << nn_test5.W[0].n_rows << " x " << nn_test5.W[0].n_cols << std::endl;
                    printf("\n");
                }
                if (!arma::approx_equal(nn_test5.W[1], nn.W[1], "both", my_tol, my_tol))
                {
                    printf("\n");
                    std::cout << "Problem with W[1] for batch 0" << std::endl;
                    std::cout << "norm is " << arma::norm(nn_test5.W[1], 1) << " vs " << arma::norm(nn.W[1], 1) << std::endl;

                    arma::mat diff = abs(nn_test5.W[1] - nn.W[1]);
                    std::cout << "max difference is: " << diff.max() << " at linear index: " << diff.index_max() << std::endl;
                    std::cout << "At that location W_test[1] and W[1] are: " << nn_test5.W[1][diff.index_max()] << " and " << nn_test5.W[1][diff.index_max()] << std::endl;
                    // std::cout << "min difference is: " << diff.min() << " at linear index: " << diff.index_min() << std::endl;
                    // std::cout << "W[1] has dimensions: " << nn_test5.W[1].n_rows << " x " << nn_test5.W[1].n_cols << std::endl;
                    printf("\n");
                }
                if (!arma::approx_equal(nn_test5.b[0], nn.b[0], "both", my_tol, my_tol))
                {
                    std::cout << "Problem with b[0]" << std::endl;
                    std::cout << arma::norm(nn_test5.b[0], 1) << " vs " << arma::norm(nn.b[0], 1) << std::endl;
                }
                if (!arma::approx_equal(nn_test5.b[1], nn.b[1], "both", my_tol, my_tol))
                {
                    std::cout << "Problem with b[1]" << std::endl;
                    std::cout << arma::norm(nn_test5.b[1], 1) << " vs " << arma::norm(nn.b[1], 1) << std::endl;
                }

                // std::cout << "nn.W[0]: " << norm(nn_test5.W[0], 1) << ", " << norm(nn.W[0], 1) << std::endl;
                // std::cout << "nn.W[1] " << norm(nn_test5.W[1], 1) << ", " << norm(nn.W[1], 1) << std::endl;
                // std::cout << "nn.b[0] " << norm(nn_test5.b[0], 1) << ", " << norm(nn.b[0], 1) << std::endl;
                // std::cout << "nn.b[1]: " << norm(nn_test5.b[1], 1) << ", " << norm(nn.b[1], 1) << std::endl;
            }

            // MPI_Barrier(MPI_COMM_WORLD);
            // if (batch == 0)
            //     exit(EXIT_FAILURE);

            // do not make any edits past here. All of this should be fine.
            if (print_every <= 0)
            {
                print_flag = batch == 0;
            }
            else
            {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if (debug && rank == 0 && print_flag)
            {
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
