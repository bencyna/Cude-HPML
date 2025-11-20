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
  const int H = 1024, W = 1024, C = 3, K = 64;
  const int FH = 3, FW = 3;
  const int N = 1; 

  size_t size_I = (size_t)N * C * H * W * sizeof(double);
  size_t size_F = (size_t)K * C * FH * FW * sizeof(double);
  size_t size_O = (size_t)N * K * H * W * sizeof(double);

  std::vector<double> I(size_I / sizeof(double));
  std::vector<double> Fh(size_F / sizeof(double));
  std::vector<double> Oh(size_O / sizeof(double), 0.0);

  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      for (int x = 0; x < H; x++)
      {
        for (int y = 0; y < W; y++)
        {
          I[((n * C + c) * H + x) * W + y] = (double)c * (x + y);
        }
      }
    }
  }


  for (int k = 0; k < K; k++)
  {
    for (int c = 0; c < C; c++)
    {
      for (int j = 0; j < FH; j++)
      {
        for (int i = 0; i < FW; i++)
        {
          double w = (double)(c + k) * (4.0 - (i + j));
          Fh[((k * C + c) * FH + j) * FW + i] = w;
        }
      }
    }
  }

  double *d_I, *d_F, *d_O;
  CUDA_CHECK(cudaMalloc(&d_I, size_I));
  CUDA_CHECK(cudaMalloc(&d_F, size_F));
  CUDA_CHECK(cudaMalloc(&d_O, size_O));

  CUDA_CHECK(cudaMemcpy(d_I, I.data(), size_I, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_F, Fh.data(), size_F, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_O, 0, size_O));

  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));

  cudnnTensorDescriptor_t xDesc, yDesc;
  cudnnFilterDescriptor_t fDesc;
  cudnnConvolutionDescriptor_t convDesc;

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&fDesc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      xDesc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_DOUBLE,
      N, C, H, W));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      yDesc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_DOUBLE,
      N, K, H, W));

  CUDNN_CHECK(cudnnSetFilter4dDescriptor(
      fDesc,
      CUDNN_DATA_DOUBLE,
      CUDNN_TENSOR_NCHW,
      K, C, FH, FW));

  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      convDesc,
      /*pad_h*/ 1, /*pad_w*/ 1,
      /*u*/ 1, /*v*/ 1,
      /*dilation_h*/ 1, /*dilation_w*/ 1,
      CUDNN_CROSS_CORRELATION, 
      CUDNN_DATA_DOUBLE));

  int nOut, cOut, hOut, wOut;
  CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
      convDesc, xDesc, fDesc, &nOut, &cOut, &hOut, &wOut));
  if (nOut != N || cOut != K || hOut != H || wOut != W)
  {
    std::cerr << "Unexpected output dims: "
              << nOut << "," << cOut << "," << hOut << "," << wOut << "\n";
    return 1;
  }


  cudnnConvolutionFwdAlgo_t algo;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
      handle,
      xDesc, fDesc, convDesc, yDesc,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
      /*memoryLimitInBytes*/ 0,
      &algo));

  size_t workspace_bytes = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      handle,
      xDesc, fDesc, convDesc, yDesc,
      algo,
      &workspace_bytes));

  void *d_workspace = nullptr;
  if (workspace_bytes > 0)
  {
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
  }

  const double alpha = 1.0;
  const double beta = 0.0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  CUDNN_CHECK(cudnnConvolutionForward(
      handle,
      &alpha,
      xDesc, d_I,
      fDesc, d_F,
      convDesc,
      algo,
      d_workspace, workspace_bytes,
      &beta,
      yDesc, d_O));

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  CUDA_CHECK(cudaMemcpy(Oh.data(), d_O, size_O, cudaMemcpyDeviceToHost));

  double checksum = 0.0;
  for (double v : Oh)
    checksum += v;


  printf("%.4f,%.3f\n", checksum, ms);

  if (d_workspace)
    cudaFree(d_workspace);
  cudaFree(d_I);
  cudaFree(d_F);
  cudaFree(d_O);

  cudnnDestroyConvolutionDescriptor(convDesc);
  cudnnDestroyFilterDescriptor(fDesc);
  cudnnDestroyTensorDescriptor(xDesc);
  cudnnDestroyTensorDescriptor(yDesc);
  cudnnDestroy(handle);

  return 0;
}
