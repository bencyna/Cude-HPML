#include <iostream>
#include <cstdlib>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./q1 <K>\n";
        return 1;
    }

    long K = std::atol(argv[1]);
    long N = K * 1000000L;

    double* A = (double*) malloc(N * sizeof(double));
    double* B = (double*) malloc(N * sizeof(double));
    double* C = (double*) malloc(N * sizeof(double));

    if (!A || !B || !C) {
        std::cerr << "Memory allocation failed\n";
        return 1;
    }

    for (long i = 0; i < N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (long i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "K = " << K << " million\n";
    std::cout << "N = " << N << " elements\n";
    std::cout << "CPU add time: " << ms << " ms\n";

    free(A);
    free(B);
    free(C);

    return 0;
}
