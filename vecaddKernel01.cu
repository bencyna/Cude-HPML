///
/// vecAddKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// By David Newman
/// Created: 2011-02-16
/// Last Modified: 2025-11-05 BC
///
/// This Kernel adds two Vectors A and B in C on GPU
/// using coalesced memory access.
/// 

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    #pragma unroll
    for (int k = 0; k < N; ++k) {
        const int idx = k * total_threads + gid;
        C[idx] = A[idx] + B[idx];
    }
}