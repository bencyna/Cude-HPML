#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// error check
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

#define FILTER_H 3
#define FILTER_W 3

__global__ void conv2d_tiled(
    const double *__restrict__ paddedInput,
    const double *__restrict__ filters,
    double *__restrict__ output,
    int H, int W, int C, int K)
{
  int threadCol = threadIdx.x;
  int threadRow = threadIdx.y;

  int outCol = blockIdx.x * TILE_SIZE + threadCol;
  int outRow = blockIdx.y * TILE_SIZE + threadRow;
  int filterIndex = blockIdx.z;

  const int paddedH = H + 2;
  const int paddedW = W + 2;

  int tileInputRow0 = blockIdx.y * TILE_SIZE;
  int tileInputCol0 = blockIdx.x * TILE_SIZE;

  __shared__ double tileInput[C][TILE_SIZE + 2][TILE_SIZE + 2];

  for (int channel = 0; channel < C; channel++)
  {
    for (int localRow = threadRow; localRow < TILE_SIZE + 2; localRow += blockDim.y)
    {
      for (int localCol = threadCol; localCol < TILE_SIZE + 2; localCol += blockDim.x)
      {

        int globalRow = tileInputRow0 + localRow;
        int globalCol = tileInputCol0 + localCol;

        double value = 0.0;
        if (globalRow >= 0 && globalRow < paddedH &&
            globalCol >= 0 && globalCol < paddedW)
        {
          value = paddedInput[(channel * paddedH + globalRow) * paddedW + globalCol];
        }

        tileInput[channel][localRow][localCol] = value;
      }
    }
  }

  __syncthreads();

  if (outRow >= H || outCol >= W)
    return;

  int localTileRow = outRow - tileInputRow0;
  int localTileCol = outCol - tileInputCol0;

  double accumulatedValue = 0.0;

  for (int channel = 0; channel < C; channel++)
  {
    for (int fh = 0; fh < FILTER_H; fh++)
    {
      for (int fw = 0; fw < FILTER_W; fw++)
      {

        int filterRow = FILTER_H - 1 - fh;
        int filterCol = FILTER_W - 1 - fw;

        double filterValue =
            filters[((filterIndex * C + channel) * FILTER_H + filterRow) * FILTER_W + filterCol];

        double inputValue =
            tileInput[channel][localTileRow + fh][localTileCol + fw];

        accumulatedValue += filterValue * inputValue;
      }
    }
  }

  output[(filterIndex * H + outRow) * W + outCol] = accumulatedValue;
}

int main()
{

  const int H = 1024, W = 1024, C = 3, K = 64;
  const int paddedH = H + 2, paddedW = W + 2;

  size_t bytesInput = (size_t)C * H * W * sizeof(double);
  size_t bytesPaddedInput = (size_t)C * paddedH * paddedW * sizeof(double);
  size_t bytesFilters = (size_t)K * C * FILTER_H * FILTER_W * sizeof(double);
  size_t bytesOutput = (size_t)K * H * W * sizeof(double);

  std::vector<double> input(bytesInput / sizeof(double));
  std::vector<double> inputPadded(bytesPaddedInput / sizeof(double), 0.0);
  std::vector<double> filterValues(bytesFilters / sizeof(double));
  std::vector<double> outputHost(bytesOutput / sizeof(double), 0.0);

  for (int c = 0; c < C; c++)
    for (int x = 0; x < H; x++)
      for (int y = 0; y < W; y++)
        input[(c * H + x) * W + y] = (double)c * (x + y);

  for (int c = 0; c < C; c++)
    for (int x = 0; x < H; x++)
      for (int y = 0; y < W; y++)
        inputPadded[(c * paddedH + (x + 1)) * paddedW + (y + 1)] =
            input[(c * H + x) * W + y];

  for (int k = 0; k < K; k++)
    for (int c = 0; c < C; c++)
      for (int fh = 0; fh < FILTER_H; fh++)
        for (int fw = 0; fw < FILTER_W; fw++)
          filterValues[((k * C + c) * FILTER_H + fh) * FILTER_W + fw] =
              (double)(c + k) * (fh + fw);

  double *d_paddedInput, *d_filters, *d_output;
  CUDA_CHECK(cudaMalloc(&d_paddedInput, bytesPaddedInput));
  CUDA_CHECK(cudaMalloc(&d_filters, bytesFilters));
  CUDA_CHECK(cudaMalloc(&d_output, bytesOutput));

  CUDA_CHECK(cudaMemcpy(d_paddedInput, inputPadded.data(), bytesPaddedInput, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_filters, filterValues.data(), bytesFilters, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0, bytesOutput));

  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim(
      (W + TILE_SIZE - 1) / TILE_SIZE,
      (H + TILE_SIZE - 1) / TILE_SIZE,
      K);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  conv2d_tiled<<<gridDim, blockDim>>>(
      d_paddedInput, d_filters, d_output, H, W, C, K);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedMs = 0.0f;
  cudaEventElapsedTime(&elapsedMs, start, stop);

  CUDA_CHECK(cudaMemcpy(outputHost.data(), d_output, bytesOutput, cudaMemcpyDeviceToHost));

  double checksum = 0.0;
  for (double value : outputHost)
    checksum += value;

  printf("%.4f,%.3f\n", checksum, elapsedMs);

  cudaFree(d_paddedInput);
  cudaFree(d_filters);
  cudaFree(d_output);

  return 0;
}
