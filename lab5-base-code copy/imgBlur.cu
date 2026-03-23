///////////////////////////////////////////////////////////////////////////////
// ENSC 453/894 Lab 5 — CUDA Image Blur (Optimized)
// =========================================================================
// OPTIMIZATIONS APPLIED:
//   1. Pinned host memory (cudaMallocHost) for faster CPU-GPU DMA transfers
//   2. Shared memory tiling with separable row-sum / column-sum computation
//      reuse — reduces per-pixel work from O((2R+1)^2) to O(2*(2R+1))
//   3. cudaFree(0) warmup to exclude CUDA context init from timing
//
// Shared memory strategy (3-phase separable box blur):
//   Phase 1: Load tile of (OUTPUT_DIM + 2*BLUR_SIZE)^2 pixels into shared
//            memory.  Out-of-bounds pixels are 0-padded.
//   Phase 2: For each tile row, compute horizontal sums of (2*BLUR_SIZE+1)
//            consecutive elements at each output-column position.  These
//            "row sums" are stored in shared memory and reused vertically.
//   Phase 3: For each output pixel, accumulate (2*BLUR_SIZE+1) row sums
//            vertically, then divide by the analytically computed count of
//            valid neighbours (avoids a second shared-memory count array).
//
// Tile multiplier TILE_I:
//   Each block of BLOCK_SIZE x BLOCK_SIZE threads outputs a region of
//   (TILE_I * BLOCK_SIZE)^2 pixels.  Each thread handles TILE_I^2 pixels.
//   Larger TILE_I increases reuse relative to halo overhead but needs more
//   shared memory and may reduce occupancy.
//
//   TILE_I=1 => output 16x16, tile 58x58, smem ~17 KB
//   TILE_I=2 => output 32x32, tile 74x74, smem ~31 KB  (default)
//   TILE_I=3 => output 48x48, tile 90x90, smem ~49 KB
//   TILE_I=4 => output 64x64, tile 106x106, smem ~70 KB
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
#define TILE_I     2   // Tile multiplier — change to 1, 3, or 4 to benchmark

///////////////////////////////////////////////////////////////////////////////
// Optimised blur kernel — shared memory + separable row/column sum reuse
///////////////////////////////////////////////////////////////////////////////
__global__ void blurKernelShared(float *out, float *in,
                                 int width, int height) {

  const int OUTPUT_DIM = TILE_I * BLOCK_SIZE;            // output region side
  const int TILE_DIM   = OUTPUT_DIM + 2 * BLUR_SIZE;     // tile side (with halo)

  // Dynamic shared memory:
  //   tile[TILE_DIM * TILE_DIM]           — input pixels (0-padded halo)
  //   rowSums[TILE_DIM * OUTPUT_DIM]      — horizontal partial sums
  extern __shared__ float smem[];
  float *tile    = smem;
  float *rowSums = smem + TILE_DIM * TILE_DIM;

  // Top-left corner of this block's output region in global coords
  int outStartX = blockIdx.x * OUTPUT_DIM;
  int outStartY = blockIdx.y * OUTPUT_DIM;

  // Top-left corner of the tile (halo extends BLUR_SIZE before output)
  int tileStartX = outStartX - BLUR_SIZE;
  int tileStartY = outStartY - BLUR_SIZE;

  int tid        = threadIdx.y * BLOCK_SIZE + threadIdx.x;
  int numThreads = BLOCK_SIZE * BLOCK_SIZE;   // 256

  // ----------------------------------------------------------------
  // Phase 1 — Cooperatively load the tile from global memory
  // ----------------------------------------------------------------
  int totalTile = TILE_DIM * TILE_DIM;
  for (int idx = tid; idx < totalTile; idx += numThreads) {
    int ty = idx / TILE_DIM;
    int tx = idx % TILE_DIM;
    int gx = tileStartX + tx;
    int gy = tileStartY + ty;

    tile[idx] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                    ? in[gy * width + gx]
                    : 0.0f;       // 0-pad out-of-bounds
  }
  __syncthreads();

  // ----------------------------------------------------------------
  // Phase 2 — Compute horizontal (row-wise) partial sums
  //   rowSums[row][ocol] = sum of tile[row][ocol .. ocol+2*BLUR_SIZE]
  //   Each sum spans exactly (2*BLUR_SIZE+1) elements.
  // ----------------------------------------------------------------
  int totalRowSums = TILE_DIM * OUTPUT_DIM;
  for (int idx = tid; idx < totalRowSums; idx += numThreads) {
    int row  = idx / OUTPUT_DIM;    // tile row   [0 .. TILE_DIM)
    int ocol = idx % OUTPUT_DIM;    // output col [0 .. OUTPUT_DIM)

    float sum = 0.0f;
    int base = row * TILE_DIM + ocol;
    for (int k = 0; k <= 2 * BLUR_SIZE; k++) {
      sum += tile[base + k];
    }
    rowSums[idx] = sum;
  }
  __syncthreads();

  // ----------------------------------------------------------------
  // Phase 3 — Accumulate row sums vertically and write output
  //   Each thread computes TILE_I x TILE_I output pixels.
  // ----------------------------------------------------------------
  for (int ii = 0; ii < TILE_I; ii++) {
    for (int jj = 0; jj < TILE_I; jj++) {
      int localRow = threadIdx.y + ii * BLOCK_SIZE;
      int localCol = threadIdx.x + jj * BLOCK_SIZE;

      int globalRow = outStartY + localRow;
      int globalCol = outStartX + localCol;

      if (globalRow < height && globalCol < width) {
        // Sum (2*BLUR_SIZE+1) row sums vertically
        float sum = 0.0f;
        int base = localRow * OUTPUT_DIM + localCol;
        for (int k = 0; k <= 2 * BLUR_SIZE; k++) {
          sum += rowSums[base + k * OUTPUT_DIM];
        }

        // Analytically compute the number of valid neighbours
        // (the valid neighbourhood is always a rectangle clamped to image bounds)
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
// main
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

  // ---- Pinned host memory for faster DMA transfers ----
  float *pinnedInput, *pinnedOutput;
  wbCheck(cudaMallocHost((void **)&pinnedInput,  numBytes));
  wbCheck(cudaMallocHost((void **)&pinnedOutput, numBytes));
  memcpy(pinnedInput, hostInputImageData, numBytes);

  // ---- Device memory ----
  wbCheck(cudaMalloc((void **)&deviceInputImageData,  numBytes));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, numBytes));

  // ---- Grid / block configuration ----
  const int OUTPUT_DIM = TILE_I * BLOCK_SIZE;
  const int TILE_DIM   = OUTPUT_DIM + 2 * BLUR_SIZE;

  int numBlocksX = (imageWidth  + OUTPUT_DIM - 1) / OUTPUT_DIM;
  int numBlocksY = (imageHeight + OUTPUT_DIM - 1) / OUTPUT_DIM;
  dim3 dimGrid(numBlocksX, numBlocksY);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  // Shared memory: tile array + row-sums array
  size_t sharedMemBytes = (size_t)(TILE_DIM * TILE_DIM
                                 + TILE_DIM * OUTPUT_DIM) * sizeof(float);

  // Opt in to extended shared memory if needed (Ampere supports up to 164 KB/SM)
  wbCheck(cudaFuncSetAttribute(blurKernelShared,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)sharedMemBytes));

  printf("Image:  %d x %d\n", imageWidth, imageHeight);
  printf("Config: BLOCK_SIZE=%d  TILE_I=%d  OUTPUT_DIM=%d  TILE_DIM=%d\n",
         BLOCK_SIZE, TILE_I, OUTPUT_DIM, TILE_DIM);
  printf("Grid:   %d x %d blocks   shared mem/block: %zu bytes\n",
         numBlocksX, numBlocksY, sharedMemBytes);

  timespec timer = tic();

  ////////////////////////////////////////////////
  // H->D transfer (pinned memory)
  wbCheck(cudaMemcpy(deviceInputImageData, pinnedInput,
                     numBytes, cudaMemcpyHostToDevice));

  // 10 kernel repeats for stable timing
  for (int i = 0; i < 10; i++) {
    blurKernelShared<<<dimGrid, dimBlock, sharedMemBytes>>>(
        deviceOutputImageData, deviceInputImageData,
        imageWidth, imageHeight);
    wbCheck(cudaGetLastError());
  }

  wbCheck(cudaDeviceSynchronize());

  // D->H transfer (pinned memory)
  wbCheck(cudaMemcpy(pinnedOutput, deviceOutputImageData,
                     numBytes, cudaMemcpyDeviceToHost));
  memcpy(hostOutputImageData, pinnedOutput, numBytes);
  ////////////////////////////////////////////////

  toc(&timer, "GPU execution time (including data transfer) in seconds");

  // ---- Correctness check against golden output ----
  for (int i = 0; i < imageHeight; i++) {
    for (int j = 0; j < imageWidth; j++) {
      float gold = goldOutputImageData[i * imageWidth + j];
      float outv = hostOutputImageData[i * imageWidth + j];

      if (fabs(gold) > 1e-6f) {
        if (fabs(outv - gold) / fabs(gold) > 0.01f) {
          printf("Incorrect output at pixel (%d, %d): gold = %f, got = %f\n",
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
          printf("Incorrect output at pixel (%d, %d): gold = %f, got = %f\n",
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
