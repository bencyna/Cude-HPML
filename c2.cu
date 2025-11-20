#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}

#define TILE 16
#define FH 3
#define FW 3

// --------------------------------------
// Tiled convolution kernel (16x16 tile)
// --------------------------------------
__global__ void conv2d_tiled(
    const double* __restrict__ I0,   // (C, H+2, W+2)
    const double* __restrict__ F,    // (K, C, 3, 3)
    double* __restrict__ O,          // (K, H, W)
    int H, int W, int C, int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int out_y = blockIdx.x * TILE + tx; // width index  (0..W-1)
    int out_x = blockIdx.y * TILE + ty; // height index (0..H-1)
    int k     = blockIdx.z;             // output channel

    const int H0 = H + 2;
    const int W0 = W + 2;

    // Top-left corner of this tile in padded input coordinates
    int tile_x0 = blockIdx.y * TILE;  // corresponds to x in [0..H+1]
    int tile_y0 = blockIdx.x * TILE;  // corresponds to y in [0..W+1]

    // Shared memory: C x (TILE+2) x (TILE+2)
    __shared__ double sh_I[3][TILE + 2][TILE + 2];  // C=3 fixed by problem

    // ---------------------------
    // Load I0 tile into shared
    // ---------------------------
    for (int c = 0; c < C; c++) {
        // Cooperative load: iterate over tile indices with stride blockDim
        for (int sx = ty; sx < TILE + 2; sx += blockDim.y) {
            for (int sy = tx; sy < TILE + 2; sy += blockDim.x) {

                int gx = tile_x0 + sx;  // global x in padded
                int gy = tile_y0 + sy;  // global y in padded

                double val = 0.0;
                if (gx >= 0 && gx < H0 && gy >= 0 && gy < W0) {
                    val = I0[(c * H0 + gx) * W0 + gy];
                }
                sh_I[c][sx][sy] = val;
            }
        }
    }

    __syncthreads();

    // Threads that map outside the output image do no work
    if (out_x >= H || out_y >= W) {
        return;
    }

    // Local coords in tile for the *top-left* of the 3x3 window
    // For output pixel at (out_x, out_y), its padded coordinates start at (out_x, out_y)
    // In tile coordinates: (out_x - tile_x0, out_y - tile_y0)
    int lx0 = out_x - tile_x0;
    int ly0 = out_y - tile_y0;

    double acc = 0.0;

    // Convolution over C, FH, FW
    for (int c = 0; c < C; c++) {
        for (int j = 0; j < FH; j++) {
            for (int i = 0; i < FW; i++) {

                int fi = FW - 1 - i;   // FW-1-i
                int fj = FH - 1 - j;   // FH-1-j

                double fval = F[((k*C + c)*FH + fj)*FW + fi];

                // I0[c, out_x + j, out_y + i] mapped into tile:
                int sx = lx0 + j;
                int sy = ly0 + i;

                double ival = sh_I[c][sx][sy];

                acc += fval * ival;
            }
        }
    }

    // Write result
    O[(k*H + out_x)*W + out_y] = acc;
}


// --------------------------------------
// MAIN
// --------------------------------------
int main() {
    const int H = 1024, W = 1024, C = 3, K = 64;
    const int H0 = H + 2, W0 = W + 2;

    size_t size_I0 = (size_t)C * H0 * W0 * sizeof(double);
    size_t size_I  = (size_t)C * H  * W  * sizeof(double);
    size_t size_F  = (size_t)K * C * FH * FW * sizeof(double);
    size_t size_O  = (size_t)K * H * W * sizeof(double);

    // Host buffers
    std::vector<double> I(size_I / sizeof(double));
    std::vector<double> I0(size_I0 / sizeof(double), 0.0);
    std::vector<double> Fh(size_F / sizeof(double));
    std::vector<double> Oh(size_O / sizeof(double), 0.0);

    // I[c,x,y] = c*(x+y)
    for (int c = 0; c < C; c++)
        for (int x = 0; x < H; x++)
            for (int y = 0; y < W; y++)
                I[(c*H + x)*W + y] = (double)c * (x + y);

    // Pad into I0 (P=1)
    for (int c = 0; c < C; c++)
        for (int x = 0; x < H; x++)
            for (int y = 0; y < W; y++)
                I0[(c*H0 + (x+1))*W0 + (y+1)] = I[(c*H + x)*W + y];

    // F[k,c,i,j] = (c+k)*(i+j)
    for (int k = 0; k < K; k++)
        for (int c = 0; c < C; c++)
            for (int j = 0; j < FH; j++)
                for (int i = 0; i < FW; i++)
                    Fh[((k*C + c)*FH + j)*FW + i] =
                        (double)(c + k) * (i + j);

    // Device memory
    double *d_I0, *d_F, *d_O;
    CUDA_CHECK(cudaMalloc(&d_I0, size_I0));
    CUDA_CHECK(cudaMalloc(&d_F,  size_F));
    CUDA_CHECK(cudaMalloc(&d_O,  size_O));

    CUDA_CHECK(cudaMemcpy(d_I0, I0.data(), size_I0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F,  Fh.data(), size_F,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, size_O));

    // Launch config: 16x16 threads, tiles of size 16x16
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE,
              (H + TILE - 1) / TILE,
              K);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    conv2d_tiled<<<grid, block>>>(d_I0, d_F, d_O, H, W, C, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(Oh.data(), d_O, size_O, cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (double v : Oh) checksum += v;

    // Required format:
    // C2_checksum,C2_execution_time
    printf("%.4f,%.3f\n", checksum, ms);

    cudaFree(d_I0);
    cudaFree(d_F);
    cudaFree(d_O);

    return 0;
}
