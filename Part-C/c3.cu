#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>

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

#define CUDNN_CHECK(ans)                    \
  {                                         \
    cudnnAssert((ans), __FILE__, __LINE__); \
  }
inline void cudnnAssert(cudnnStatus_t status, const char *file, int line)
{
  if (status != CUDNN_STATUS_SUCCESS)
  {
    std::cerr << "cuDNN Error: " << cudnnGetErrorString(status)
              << " at " << file << ":" << line << std::endl;
    exit(1);
  }
}

int main()
{
  const int H = 1024;
  const int W = 1024;
  const int C = 3;
  const int K = 64;
  const int FH = 3;
  const int FW = 3;
  const int N = 1;

  size_t bytesInput = (size_t)N * C * H * W * sizeof(double);
  size_t bytesFilter = (size_t)K * C * FH * FW * sizeof(double);
  size_t bytesOutput = (size_t)N * K * H * W * sizeof(double);

  std::vector<double> inputHost(bytesInput / sizeof(double));
  std::vector<double> filterHost(bytesFilter / sizeof(double));
  std::vector<double> outputHost(bytesOutput / sizeof(double), 0.0);

  // init for input: I[n,c,x,y] = c*(x+y)
  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      for (int x = 0; x < H; x++)
      {
        for (int y = 0; y < W; y++)
        {
          inputHost[((n * C + c) * H + x) * W + y] =
              (double)c * (x + y);
        }
      }
    }
  }

  for (int k = 0; k < K; k++)
  {
    for (int c = 0; c < C; c++)
    {
      for (int fh = 0; fh < FH; fh++)
      {
        for (int fw = 0; fw < FW; fw++)
        {
          double weight = (double)(c + k) * (4.0 - (fh + fw));
          filterHost[((k * C + c) * FH + fh) * FW + fw] = weight;
        }
      }
    }
  }

  double *d_input = nullptr;
  double *d_filter = nullptr;
  double *d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, bytesInput));
  CUDA_CHECK(cudaMalloc(&d_filter, bytesFilter));
  CUDA_CHECK(cudaMalloc(&d_output, bytesOutput));

  CUDA_CHECK(cudaMemcpy(d_input, inputHost.data(), bytesInput, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_filter, filterHost.data(), bytesFilter, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0, bytesOutput));

  cudnnHandle_t cudnnHandle;
  CUDNN_CHECK(cudnnCreate(&cudnnHandle));

  cudnnTensorDescriptor_t inputDesc, outputDesc;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      inputDesc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_DOUBLE,
      N, C, H, W));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      outputDesc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_DOUBLE,
      N, K, H, W));

  CUDNN_CHECK(cudnnSetFilter4dDescriptor(
      filterDesc,
      CUDNN_DATA_DOUBLE,
      CUDNN_TENSOR_NCHW,
      K, C, FH, FW));

  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      convDesc,
      1, 1,
      1, 1,
      1, 1,
      CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_DOUBLE));

  // check
  int outN, outC, outH, outW;
  CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
      convDesc, inputDesc, filterDesc, &outN, &outC, &outH, &outW));
  if (outN != N || outC != K || outH != H || outW != W)
  {
    std::cerr << "Unexpected output dims: "
              << outN << "," << outC << "," << outH << "," << outW << "\n";
    return 1;
  }

  cudnnConvolutionFwdAlgo_t fwdAlgo =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  size_t workspaceBytes = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      cudnnHandle,
      inputDesc, filterDesc, convDesc, outputDesc,
      fwdAlgo,
      &workspaceBytes));

  void *d_workspace = nullptr;
  if (workspaceBytes > 0)
  {
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceBytes));
  }

  const double alpha = 1.0;
  const double beta = 0.0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  CUDNN_CHECK(cudnnConvolutionForward(
      cudnnHandle,
      &alpha,
      inputDesc, d_input,
      filterDesc, d_filter,
      convDesc,
      fwdAlgo,
      d_workspace, workspaceBytes,
      &beta,
      outputDesc, d_output));

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedMs = 0.0f;
  cudaEventElapsedTime(&elapsedMs, start, stop);

  CUDA_CHECK(cudaMemcpy(outputHost.data(), d_output, bytesOutput, cudaMemcpyDeviceToHost));

  double checksum = 0.0;
  for (double v : outputHost)
  {
    checksum += v;
  }

  printf("%.4f,%.3f\n", checksum, elapsedMs);

  if (d_workspace)
    cudaFree(d_workspace);
  cudaFree(d_input);
  cudaFree(d_filter);
  cudaFree(d_output);

  cudnnDestroyConvolutionDescriptor(convDesc);
  cudnnDestroyFilterDescriptor(filterDesc);
  cudnnDestroyTensorDescriptor(inputDesc);
  cudnnDestroyTensorDescriptor(outputDesc);
  cudnnDestroy(cudnnHandle);

  return 0;
}
