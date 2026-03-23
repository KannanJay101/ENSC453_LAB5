///////////////////////////////////////////////////////////////////////////////
// =========================================================================
// VARIATION: SHARED MEMORY OPTIMIZATION ONLY
// =========================================================================
//
// What CHANGED from baseline:
//   - Kernel uses shared memory tiling with separable row-sum / column-sum
//     computation reuse (3-phase approach)
//   - TILE_I parameter controls output region size
//
// What is IDENTICAL to baseline:
//   - Pageable host memory (NO cudaMallocHost / pinned memory)
//   - NO cudaFree(0) warmup
//   - 10 kernel repeats for timing stability
//   - cudaMemcpy directly from hostInputImageData
//
// Compare execution time against baseline (1.265s) to isolate the
// shared memory speedup:  speedup = (1.265 - this_time) / 1.265 * 100
///////////////////////////////////////////////////////////////////////////////

#include "libwb/wb.h"
#include "my_timer.h"
#include <math.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLUR_SIZE  21
#define BLOCK_SIZE 16
#define TILE_I     2

///////////////////////////////////////////////////////////////////////////////
// Optimised blur kernel — shared memory + separable row/column sum reuse
///////////////////////////////////////////////////////////////////////////////
__global__ void blurKernelShared(float *out, float *in,
                                 int width, int height) {

  const int OUTPUT_DIM = TILE_I * BLOCK_SIZE;
  const int TILE_DIM   = OUTPUT_DIM + 2 * BLUR_SIZE;

  extern __shared__ float smem[];
  float *tile    = smem;
  float *rowSums = smem + TILE_DIM * TILE_DIM;

  int outStartX = blockIdx.x * OUTPUT_DIM;
  int outStartY = blockIdx.y * OUTPUT_DIM;

  int tileStartX = outStartX - BLUR_SIZE;
  int tileStartY = outStartY - BLUR_SIZE;

  int tid        = threadIdx.y * BLOCK_SIZE + threadIdx.x;
  int numThreads = BLOCK_SIZE * BLOCK_SIZE;

  // Phase 1 — Load tile from global memory
  int totalTile = TILE_DIM * TILE_DIM;
  for (int idx = tid; idx < totalTile; idx += numThreads) {
    int ty = idx / TILE_DIM;
    int tx = idx % TILE_DIM;
    int gx = tileStartX + tx;
    int gy = tileStartY + ty;

    tile[idx] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                    ? in[gy * width + gx]
                    : 0.0f;
  }
  __syncthreads();

  // Phase 2 — Compute horizontal (row-wise) partial sums
  int totalRowSums = TILE_DIM * OUTPUT_DIM;
  for (int idx = tid; idx < totalRowSums; idx += numThreads) {
    int row  = idx / OUTPUT_DIM;
    int ocol = idx % OUTPUT_DIM;

    float sum = 0.0f;
    int base = row * TILE_DIM + ocol;
    for (int k = 0; k <= 2 * BLUR_SIZE; k++) {
      sum += tile[base + k];
    }
    rowSums[idx] = sum;
  }
  __syncthreads();

  // Phase 3 — Accumulate row sums vertically and write output
  for (int ii = 0; ii < TILE_I; ii++) {
    for (int jj = 0; jj < TILE_I; jj++) {
      int localRow = threadIdx.y + ii * BLOCK_SIZE;
      int localCol = threadIdx.x + jj * BLOCK_SIZE;

      int globalRow = outStartY + localRow;
      int globalCol = outStartX + localCol;

      if (globalRow < height && globalCol < width) {
        float sum = 0.0f;
        int base = localRow * OUTPUT_DIM + localCol;
        for (int k = 0; k <= 2 * BLUR_SIZE; k++) {
          sum += rowSums[base + k * OUTPUT_DIM];
        }

        int x0 = max(0, globalCol - BLUR_SIZE);
        int x1 = min(width  - 1, globalCol + BLUR_SIZE);
        int y0 = max(0, globalRow - BLUR_SIZE);
        int y1 = min(height - 1, globalRow + BLUR_SIZE);
        int count = (x1 - x0 + 1) * (y1 - y0 + 1);

        out[globalRow * width + globalCol] = sum / count;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// main — IDENTICAL to baseline (pageable memory, no warmup)
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

  // NO cudaFree(0) warmup — same as baseline

  const int OUTPUT_DIM = TILE_I * BLOCK_SIZE;
  const int TILE_DIM   = OUTPUT_DIM + 2 * BLUR_SIZE;

  int numBlocksX = (imageWidth  + OUTPUT_DIM - 1) / OUTPUT_DIM;
  int numBlocksY = (imageHeight + OUTPUT_DIM - 1) / OUTPUT_DIM;
  dim3 dimGrid(numBlocksX, numBlocksY);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  size_t sharedMemBytes = (size_t)(TILE_DIM * TILE_DIM
                                 + TILE_DIM * OUTPUT_DIM) * sizeof(float);

  wbCheck(cudaFuncSetAttribute(blurKernelShared,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)sharedMemBytes));

  size_t numBytes = imageWidth * imageHeight * sizeof(float);

  timespec timer = tic();

  ////////////////////////////////////////////////
  // Pageable memory transfers — same as baseline

  wbCheck(cudaMalloc((void **)&deviceInputImageData, numBytes));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, numBytes));

  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData,
                     numBytes, cudaMemcpyHostToDevice));

  // 10 kernel repeats — same as baseline
  for (int i = 0; i < 10; i++) {
    blurKernelShared<<<dimGrid, dimBlock, sharedMemBytes>>>(
        deviceOutputImageData, deviceInputImageData,
        imageWidth, imageHeight);
    wbCheck(cudaGetLastError());
  }

  wbCheck(cudaDeviceSynchronize());

  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData,
                     numBytes, cudaMemcpyDeviceToHost));

  ////////////////////////////////////////////////

  toc(&timer, "GPU execution time (including data transfer) in seconds");

  // Correctness check
  for (int i = 0; i < imageHeight; i++) {
    for (int j = 0; j < imageWidth; j++) {
      float gold = goldOutputImageData[i * imageWidth + j];
      float outv = hostOutputImageData[i * imageWidth + j];

      if (fabs(gold) > 1e-6f) {
        if (fabs(outv - gold) / fabs(gold) > 0.01f) {
          printf("Incorrect output image at pixel (%d, %d): goldOutputImage = %f, hostOutputImage = %f\n",
                 i, j, gold, outv);
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

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);
  wbImage_delete(goldImage);

  return 0;
}
