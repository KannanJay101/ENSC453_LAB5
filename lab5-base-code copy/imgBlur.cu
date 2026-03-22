///////////////////////////////////////////////////////////////////////////////
// =========================================================================
// VARIATION: DATA TRANSFER OPTIMIZATION ONLY
// =========================================================================
//
// What CHANGED from baseline:
//   - Pinned host memory (cudaMallocHost) for faster CPU-GPU DMA transfers
//   - cudaFree(0) warmup before timing to exclude CUDA context init
//
// What is IDENTICAL to baseline:
//   - Naive blur kernel (reads directly from global memory, NO shared memory)
//   - 10 kernel repeats for timing stability
//   - Same grid/block dimensions (BLOCK_SIZE x BLOCK_SIZE = 16 x 16)
//
// Tile size:         N/A     (no tiling)
// Threads per block: 256     (16 x 16)
//
// Compare execution time against baseline (1.265s) to isolate the
// data transfer speedup:  speedup = (1.265 - this_time) / 1.265 * 100
///////////////////////////////////////////////////////////////////////////////

#include "libwb/wb.h"
#include "my_timer.h"
#include <math.h>

#define wbCheck(stmt)                                                   \
  do {                                                                  \
    cudaError_t err = stmt;                                             \
    if (err != cudaSuccess) {                                           \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
      return -1;                                                        \
    }                                                                   \
  } while (0)

#define BLUR_SIZE 21
#define BLOCK_SIZE 16

///////////////////////////////////////////////////////////////////////////////
// Baseline box blur kernel (global memory, no tiling) — UNCHANGED
// Each thread computes one output pixel by averaging all valid pixels
// in a (BLUR_SIZE x BLUR_SIZE) neighbourhood centred on (x, y).
///////////////////////////////////////////////////////////////////////////////
__global__ void blurKernel(float *out, float *in, int width, int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  float sum = 0.0f;
  int count = 0;

  for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++) {
    for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++) {
      int nx = x + j;
      int ny = y + i;

      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        sum += in[ny * width + nx];
        count++;
      }
    }
  }

  out[y * width + x] = sum / count;
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  wbImage_t goldImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *goldOutputImageData;

  args = wbArg_read(argc, argv);

  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage = wbImport(inputImageFile);

  char *goldImageFile = argv[2];
  goldImage = wbImport(goldImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  goldOutputImageData = wbImage_getData(goldImage);

  // Force CUDA context initialization before timing
  wbCheck(cudaFree(0));

  size_t numBytes = imageWidth * imageHeight * sizeof(float);

  // Pinned memory for faster CPU-GPU DMA transfers
  float *pinnedInput, *pinnedOutput;
  wbCheck(cudaMallocHost((void **)&pinnedInput, numBytes));
  wbCheck(cudaMallocHost((void **)&pinnedOutput, numBytes));
  memcpy(pinnedInput, hostInputImageData, numBytes);

  wbCheck(cudaMalloc((void **)&deviceInputImageData, numBytes));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, numBytes));

  int numBlocksX = (imageWidth + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int numBlocksY = (imageHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(numBlocksX, numBlocksY);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  timespec timer = tic();

  ////////////////////////////////////////////////
  // Pinned memory transfer (optimization) instead of pageable
  wbCheck(cudaMemcpy(deviceInputImageData, pinnedInput,
                     numBytes, cudaMemcpyHostToDevice));

  // 10 kernel repeats — same as baseline
  for (int i = 0; i < 10; i++) {
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData,
                                      deviceInputImageData,
                                      imageWidth, imageHeight);
    wbCheck(cudaGetLastError());
  }

  wbCheck(cudaDeviceSynchronize());

  // Pinned memory transfer back
  wbCheck(cudaMemcpy(pinnedOutput, deviceOutputImageData,
                     numBytes, cudaMemcpyDeviceToHost));
  memcpy(hostOutputImageData, pinnedOutput, numBytes);

  ///////////////////////////////////////////////////////

  toc(&timer, "GPU execution time (including data transfer) in seconds");

  // Correctness check against golden output
  for (int i = 0; i < imageHeight; i++) {
    for (int j = 0; j < imageWidth; j++) {
      float gold = goldOutputImageData[i * imageWidth + j];
      float outv = hostOutputImageData[i * imageWidth + j];

      if (fabs(gold) > 1e-6f) {
        if (fabs(outv - gold) / fabs(gold) > 0.01f) {
          printf("Incorrect output image at pixel (%d, %d): goldOutputImage = %f, hostOutputImage = %f\n",
                 i, j, gold, outv);
          cudaFreeHost(pinnedInput);
          cudaFreeHost(pinnedOutput);
          cudaFree(deviceInputImageData);
          cudaFree(deviceOutputImageData);
          wbImage_delete(outputImage);
          wbImage_delete(inputImage);
          wbImage_delete(goldImage);
          return -1;
        }
      } else {
        if (fabs(outv - gold) > 1e-6f) {
          printf("Incorrect output image at pixel (%d, %d): goldOutputImage = %f, hostOutputImage = %f\n",
                 i, j, gold, outv);
          cudaFreeHost(pinnedInput);
          cudaFreeHost(pinnedOutput);
          cudaFree(deviceInputImageData);
          cudaFree(deviceOutputImageData);
          wbImage_delete(outputImage);
          wbImage_delete(inputImage);
          wbImage_delete(goldImage);
          return -1;
        }
      }
    }
  }

  printf("Correct output image!\n");

  cudaFreeHost(pinnedInput);
  cudaFreeHost(pinnedOutput);
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);
  wbImage_delete(goldImage);

  return 0;
}
