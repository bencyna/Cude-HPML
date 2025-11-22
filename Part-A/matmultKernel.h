/// matmultKernel.h
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-19 DVN
///
/// Kernels defined with this header must 
/// multiply two matrices using CUDA: A x B = C
///

#ifndef __MMKERNEL__
#define __MMKERNEL__

// Defines for size of thread block and data computed by a thread block
#define BLOCK_SIZE 16

// FOOTPRINT_SIZE can be overridden at compile time (e.g. -DFOOTPRINT_SIZE=32)
#ifndef FOOTPRINT_SIZE
#define FOOTPRINT_SIZE BLOCK_SIZE
#endif

// The type Matrix is really a MATRIX DESCRIPTOR. 
// Matrices are stored in row major order:
//   M[row,col] = *(M.elements + row * M.stride + col)
//
// A submatrix is not copied but represented as a descriptor that
// points into a larger matrix. The 'stride' is the width (in elements)
// from one row of the large matrix to the next row.

typedef struct {
    int   width;
    int   height;
    int   stride;
    float *elements;
} Matrix;

// Forward declaration of the kernel function that performs the work.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C);

#endif // __MMKERNEL__
