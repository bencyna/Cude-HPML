///
/// matmultKernel01.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments.
///

#include "matmultKernel.h"

#ifndef FOOTPRINT_SIZE
#define FOOTPRINT_SIZE BLOCK_SIZE
#endif

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;

  int block_row = blockIdx.y * FOOTPRINT_SIZE;
  int block_col = blockIdx.x * FOOTPRINT_SIZE;

  int row0 = block_row + thread_row;
  int row1 = row0 + BLOCK_SIZE;
  int col0 = block_col + thread_col;
  int col1 = col0 + BLOCK_SIZE;

  // Shared memory tiles for A and B
  __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
  __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

  float C00 = 0.0f;
  float C01 = 0.0f;
  float C10 = 0.0f;
  float C11 = 0.0f;

  int numTiles = A.width / FOOTPRINT_SIZE;


  for (int m = 0; m < numTiles; ++m)
  {

    int kBase = m * FOOTPRINT_SIZE;

    // load a title in shared A
    int a_col0 = kBase + thread_col;
    int a_col1 = kBase + thread_col + BLOCK_SIZE;

    shared_A[thread_row][thread_col] =
        A.elements[row0 * A.stride + a_col0];
    shared_A[thread_row + BLOCK_SIZE][thread_col] =
        A.elements[row1 * A.stride + a_col0];
    shared_A[thread_row][thread_col + BLOCK_SIZE] =
        A.elements[row0 * A.stride + a_col1];
    shared_A[thread_row + BLOCK_SIZE][thread_col + BLOCK_SIZE] =
        A.elements[row1 * A.stride + a_col1];

    // load b tile into share b
    int b_row0 = kBase + thread_row;
    int b_row1 = kBase + thread_row + BLOCK_SIZE;

    shared_B[thread_row][thread_col] =
        B.elements[b_row0 * B.stride + col0];
    shared_B[thread_row + BLOCK_SIZE][thread_col] =
        B.elements[b_row1 * B.stride + col0];
    shared_B[thread_row][thread_col + BLOCK_SIZE] =
        B.elements[b_row0 * B.stride + col1];
    shared_B[thread_row + BLOCK_SIZE][thread_col + BLOCK_SIZE] =
        B.elements[b_row1 * B.stride + col1];

    __syncthreads();

    // compute c outputs from this tile
#pragma unroll
    for (int e = 0; e < FOOTPRINT_SIZE; ++e)
    {
      float a0 = shared_A[thread_row][e];
      float a1 = shared_A[thread_row + BLOCK_SIZE][e];
      float b0 = shared_B[e][thread_col];
      float b1 = shared_B[e][thread_col + BLOCK_SIZE];

      C00 += a0 * b0;
      C01 += a0 * b1;
      C10 += a1 * b0;
      C11 += a1 * b1;
    }

    __syncthreads();
  }

  // write result back into global mem
  C.elements[row0 * C.stride + col0] = C00;
  C.elements[row0 * C.stride + col1] = C01;
  C.elements[row1 * C.stride + col0] = C10;
  C.elements[row1 * C.stride + col1] = C11;
}
