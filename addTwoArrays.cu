#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " : " << cudaGetErrorString(err) << std::endl; \
        std::exit(1);                                              \
    }                                                              \
} while (0)

__global__ void add_kernel(const double* A,
                           const double* B,
                           double* C,
                           long long N)
{
    long long idx    = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (long long i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./q2 <K>\n";
        return 1;
    }

    long long K = std::atoll(argv[1]);
    long long N = K * 1000000LL;

    std::cout << "K = " << K << " million, N = " << N << " elements\n";

    double* h_A = (double*) std::malloc(N * sizeof(double));
    double* h_B = (double*) std::malloc(N * sizeof(double));
    double* h_C = (double*) std::malloc(N * sizeof(double));

    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host memory allocation failed\n";
        return 1;
    }

    for (long long i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
    }

    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, N * sizeof(double)));

    for (int scenario = 1; scenario <= 3; ++scenario) {
        int threadsPerBlock;
        int numBlocks;

        if (scenario == 1) {
            threadsPerBlock = 1;
            numBlocks = 1;
            std::cout << "\nScenario 1: 1 block, 1 thread\n";
        } else if (scenario == 2) {
            threadsPerBlock = 256;
            numBlocks = 1;
            std::cout << "\nScenario 2: 1 block, 256 threads\n";
        } else {
            threadsPerBlock = 256;

            long long blocksNeeded = (N + threadsPerBlock - 1) / threadsPerBlock;

            numBlocks = static_cast<int>(blocksNeeded);

            std::cout << "\nScenario 3: " << numBlocks
                      << " blocks, " << threadsPerBlock
                      << " threads per block (total threads ≈ N)\n";
        }


        CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, N * sizeof(double), cudaMemcpyHostToDevice));

        auto start = std::chrono::high_resolution_clock::now();

        add_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaGetLastError());     

        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C, d_C, N * sizeof(double), cudaMemcpyDeviceToHost));

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "GPU time (including memcpy) for scenario "
                  << scenario << ": " << ms << " ms\n";

        // check 
        if (N > 0) {
            if (h_C[0] != 3.0) {
                std::cerr << "Warning: unexpected result C[0] = "
                          << h_C[0] << " (expected 3.0)\n";
            }
        }
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    std::free(h_A);
    std::free(h_B);
    std::free(h_C);

    return 0;
}
