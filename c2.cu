#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans)                   \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
  {
    std::cerr << "CUDA Error: " << cudaGetErrorString(code)
              << " at " << file << ":" << line << std::endl;
    exit(1);
  }
}

#define TILE 16
#define FH 3
#define FW 3

__global__ void conv2d_tiled(
    const double *__restrict__ I0,
    const double *__restrict__ F,
    double *__restrict__ O,
    int H, int W, int C, int K)
{
  int out_y = blockIdx.x * TILE + threadIdx.x;
  int out_x = blockIdx.y * TILE + threadIdx.y;
  int k = blockIdx.z;

  __shared__ double sh_I[3][TILE + 2][TILE + 2];

  int H0 = H + 2;
  int W0 = W + 2;

  int in_y = out_y;
  int in_x = out_x;

  if (in_x < H0 && in_y < W0)
  {
    for (int c = 0; c < C; c++)
    {
      sh_I[c][threadIdx.y][threadIdx.x] =
          I0[(c * H0 + in_x) * W0 + in_y];
    }
  }

  __syncthreads();

  if (out_x >= H || out_y >= W)
    return;

  double acc = 0.0;

  for (int c = 0; c < C; c++)
  {
    for (int j = 0; j < FH; j++)
    {
      for (int i = 0; i < FW; i++)
      {
        int fi = FW - 1 - i;
        int fj = FH - 1 - j;

        double fval = F[((k * C + c) * FH + fj) * FW + fi];

        double ival = sh_I[c][threadIdx.y + j][threadIdx.x + i];

        acc += fval * ival;
      }
    }
  }

  O[(k * H + out_x) * W + out_y] = acc;
}

int main()
{
  const int H = 1024, W = 1024, C = 3, K = 64;
  const int H0 = H + 2, W0 = W + 2;

  size_t size_I0 = (size_t)C * H0 * W0 * sizeof(double);
  size_t size_I = (size_t)C * H * W * sizeof(double);
  size_t size_F = (size_t)K * C * 3 * 3 * sizeof(double);
  size_t size_O = (size_t)K * H * W * sizeof(double);

  std::vector<double> I(size_I / sizeof(double));
  std::vector<double> I0(size_I0 / sizeof(double), 0.0);
  std::vector<double> Fh(size_F / sizeof(double));
  std::vector<double> Oh(size_O / sizeof(double), 0.0);

  for (int c = 0; c < C; c++)
    for (int x = 0; x < H; x++)
      for (int y = 0; y < W; y++)
        I[(c * H + x) * W + y] = (double)c * (x + y);

  for (int c = 0; c < C; c++)
    for (int x = 0; x < H; x++)
      for (int y = 0; y < W; y++)
        I0[(c * H0 + (x + 1)) * W0 + (y + 1)] =
            I[(c * H + x) * W + y];

  for (int k = 0; k < K; k++)
    for (int c = 0; c < C; c++)
      for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
          Fh[((k * C + c) * 3 + j) * 3 + i] =
              (double)(c + k) * (i + j);

  double *d_I0, *d_F, *d_O;
  CUDA_CHECK(cudaMalloc(&d_I0, size_I0));
  CUDA_CHECK(cudaMalloc(&d_F, size_F));
  CUDA_CHECK(cudaMalloc(&d_O, size_O));

  CUDA_CHECK(cudaMemcpy(d_I0, I0.data(), size_I0, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_F, Fh.data(), size_F, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_O, 0, size_O));

  dim3 block(TILE + 2, TILE + 2);
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
  for (double v : Oh)
    checksum += v;

  printf("%.4f,%.3f\n", checksum, ms);

  cudaFree(d_I0);
  cudaFree(d_F);
  cudaFree(d_O);

  return 0;
}
