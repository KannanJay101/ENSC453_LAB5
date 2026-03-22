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
#ifndef BLUR_KERNEL_REPEATS
#define BLUR_KERNEL_REPEATS 1
#endif

// Basic shared memory tile: output region is BLOCK_SIZE x BLOCK_SIZE,
// tile includes halo of BLUR_SIZE on each side.
#define OUT_WIDTH  BLOCK_SIZE
#define OUT_HEIGHT BLOCK_SIZE
#define TILE_WIDTH  (OUT_WIDTH  + 2 * BLUR_SIZE)   // 16 + 42 = 58
#define TILE_HEIGHT (OUT_HEIGHT + 2 * BLUR_SIZE)   // 16 + 42 = 58

///////////////////////////////////////////////////////////////////////////////
// Basic shared memory blur kernel.
// Each thread block loads a tile (output region + halo) into shared memory,
// then each thread computes its output pixel by reading from shared memory
// instead of global memory. No computation reuse (no colSum).
///////////////////////////////////////////////////////////////////////////////
__global__ void blurKernel(float *out, const float *in,
                           int width, int height) {

  __shared__ float tile[TILE_HEIGHT][TILE_WIDTH];

  // Top-left corner of this block's output region in global coords
  int outX0 = blockIdx.x * OUT_WIDTH;
  int outY0 = blockIdx.y * OUT_HEIGHT;

  // Top-left corner of the tile (includes halo)
  int tileStartX = outX0 - BLUR_SIZE;
  int tileStartY = outY0 - BLUR_SIZE;

  // --- Step 1: Cooperatively load tile into shared memory ---
  int numElements = TILE_WIDTH * TILE_HEIGHT;
  int threadId    = threadIdx.y * BLOCK_SIZE + threadIdx.x;
  int numThreads  = BLOCK_SIZE * BLOCK_SIZE;

  for (int idx = threadId; idx < numElements; idx += numThreads) {
    int tileRow = idx / TILE_WIDTH;
    int tileCol = idx % TILE_WIDTH;

    int gx = tileStartX + tileCol;
    int gy = tileStartY + tileRow;

    if (gx >= 0 && gx < width && gy >= 0 && gy < height)
      tile[tileRow][tileCol] = in[gy * width + gx];
    else
      tile[tileRow][tileCol] = 0.0f;
  }

  __syncthreads();

  // --- Step 2: Each thread computes one output pixel from shared memory ---
  int outX = outX0 + threadIdx.x;
  int outY = outY0 + threadIdx.y;
  if (outX >= width || outY >= height) return;

  // Clamp blur window to image boundaries
  int x0 = max(0, outX - BLUR_SIZE);
  int x1 = min(width  - 1, outX + BLUR_SIZE);
  int y0 = max(0, outY - BLUR_SIZE);
  int y1 = min(height - 1, outY + BLUR_SIZE);

  float sum = 0.0f;
  int count = (x1 - x0 + 1) * (y1 - y0 + 1);

  // Convert global coords to tile-local coords and accumulate
  for (int gy = y0; gy <= y1; gy++) {
    for (int gx = x0; gx <= x1; gx++) {
      sum += tile[gy - tileStartY][gx - tileStartX];
    }
  }

  out[outY * width + outX] = sum / (float)count;
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

  // Use pinned memory for faster CPU-GPU transfers
  float *pinnedInput;
  float *pinnedOutput;
  wbCheck(cudaMallocHost((void **)&pinnedInput, numBytes));
  wbCheck(cudaMallocHost((void **)&pinnedOutput, numBytes));
  memcpy(pinnedInput, hostInputImageData, numBytes);

  wbCheck(cudaMalloc((void **)&deviceInputImageData, numBytes));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, numBytes));

  // Grid: one block per BLOCK_SIZE x BLOCK_SIZE output region
  int numBlocksX = (imageWidth  + OUT_WIDTH  - 1) / OUT_WIDTH;
  int numBlocksY = (imageHeight + OUT_HEIGHT - 1) / OUT_HEIGHT;
  dim3 dimGrid(numBlocksX, numBlocksY);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  timespec timer = tic();

  wbCheck(cudaMemcpy(deviceInputImageData, pinnedInput, numBytes,
                     cudaMemcpyHostToDevice));

  for (int i = 0; i < BLUR_KERNEL_REPEATS; i++) {
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData,
                                      deviceInputImageData,
                                      imageWidth, imageHeight);
    wbCheck(cudaGetLastError());
  }

  wbCheck(cudaDeviceSynchronize());

  wbCheck(cudaMemcpy(pinnedOutput, deviceOutputImageData,
                     numBytes, cudaMemcpyDeviceToHost));
  memcpy(hostOutputImageData, pinnedOutput, numBytes);

  toc(&timer, "GPU execution time (including data transfer) in seconds");

  // Correctness check against golden output
  for (int i = 0; i < imageHeight; i++) {
    for (int j = 0; j < imageWidth; j++) {
      float gold = goldOutputImageData[i * imageWidth + j];
      float outv = hostOutputImageData[i * imageWidth + j];

      if (fabs(gold) > 1e-6f) {
        if (fabs(outv - gold) / fabs(gold) > 0.01f) {
          printf("Incorrect output image at pixel (%d, %d): "
                 "goldOutputImage = %f, hostOutputImage = %f\n",
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
          printf("Incorrect output image at pixel (%d, %d): "
                 "goldOutputImage = %f, hostOutputImage = %f\n",
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
