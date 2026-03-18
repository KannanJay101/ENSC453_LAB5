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

// Tile dimensions including halo on all sides
#define TILE_WIDTH  (BLOCK_SIZE + 2 * BLUR_SIZE)  // 16 + 42 = 58
#define TILE_HEIGHT (BLOCK_SIZE + 2 * BLUR_SIZE)  // 16 + 42 = 58

///////////////////////////////////////////////////////////////////////////////
// Optimized blur kernel using shared memory with computation reuse.
//
// The optimization works in 3 steps:
//   Step 1: Cooperatively load a tile (58x58) from global memory into shared
//           memory. The tile covers the output region (16x16) plus a halo of
//           BLUR_SIZE=21 pixels on every side.
//   Step 2: Each thread computes a vertical column sum over (2*BLUR_SIZE+1)=43
//           rows for its assigned columns, storing results in colSum[].
//   Step 3: Each thread accumulates horizontally across 43 precomputed column
//           sums to get the final blurred pixel value.
//
// This reduces per-thread arithmetic from O(BLUR_SIZE^2) = 1849 additions
// down to O(BLUR_SIZE) + O(BLUR_SIZE) = 86 additions.
///////////////////////////////////////////////////////////////////////////////
__global__ void blurKernel(float *out, float *in, int width, int height) {

  // Shared memory: tile holds the input image patch, colSum holds vertical sums
  __shared__ float tile[TILE_HEIGHT][TILE_WIDTH];
  __shared__ float colSum[BLOCK_SIZE][TILE_WIDTH];

  // Output pixel coordinates in the global image
  int outX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int outY = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  // Top-left corner of the tile in global image coordinates (including halo)
  int tileStartX = blockIdx.x * BLOCK_SIZE - BLUR_SIZE;
  int tileStartY = blockIdx.y * BLOCK_SIZE - BLUR_SIZE;

  //-----------------------------------------------------------
  // Step 1: Cooperatively load the tile into shared memory.
  // The tile is 58x58 = 3364 elements but we only have 16x16 = 256 threads,
  // so each thread loads multiple elements using a strided loop.
  //-----------------------------------------------------------
  int numElements = TILE_WIDTH * TILE_HEIGHT;       // 58 * 58 = 3364
  int threadId = threadIdx.y * BLOCK_SIZE + threadIdx.x;  // Unique thread ID (0-255)
  int numThreads = BLOCK_SIZE * BLOCK_SIZE;         // 16 * 16 = 256

  for (int idx = threadId; idx < numElements; idx += numThreads) {
    // Convert flat index to 2D tile coordinates
    int tileRow = idx / TILE_WIDTH;
    int tileCol = idx % TILE_WIDTH;

    // Convert tile coordinates to global image coordinates
    int globalX = tileStartX + tileCol;
    int globalY = tileStartY + tileRow;

    // Load from global memory; use 0 for out-of-bounds pixels
    if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
      tile[tileRow][tileCol] = in[globalY * width + globalX];
    } else {
      tile[tileRow][tileCol] = 0.0f;
    }
  }

  // Wait for all threads to finish loading the tile
  __syncthreads();

  //-----------------------------------------------------------
  // Step 2: Compute vertical column sums.
  // For the output row that this thread handles (threadIdx.y),
  // sum (2*BLUR_SIZE+1) = 43 elements vertically in each column.
  // Each thread handles multiple columns using a strided loop.
  //
  // colSum[threadIdx.y][c] = tile[threadIdx.y][c] + tile[threadIdx.y+1][c]
  //                        + ... + tile[threadIdx.y + 2*BLUR_SIZE][c]
  //
  // This is the key reuse: horizontally adjacent threads in the
  // same row share the same colSum values.
  //-----------------------------------------------------------
  for (int c = threadIdx.x; c < TILE_WIDTH; c += BLOCK_SIZE) {
    float vsum = 0.0f;
    for (int dy = 0; dy <= 2 * BLUR_SIZE; dy++) {
      vsum += tile[threadIdx.y + dy][c];
    }
    colSum[threadIdx.y][c] = vsum;
  }

  // Wait for all column sums to be computed
  __syncthreads();

  //-----------------------------------------------------------
  // Step 3: Horizontal accumulation and output.
  // Each thread sums across the precomputed column sums for its
  // horizontal neighborhood, then divides by the valid pixel count.
  // Boundary handling: we need the actual count of valid pixels
  // (not including out-of-bounds zeros that were loaded as 0).
  //-----------------------------------------------------------
  if (outX >= width || outY >= height) return;

  // Compute the valid boundary of the neighborhood in global coords
  int y0 = max(0, outY - BLUR_SIZE);
  int y1 = min(height - 1, outY + BLUR_SIZE);
  int x0 = max(0, outX - BLUR_SIZE);
  int x1 = min(width - 1, outX + BLUR_SIZE);

  int validRows = y1 - y0 + 1;
  int validCols = x1 - x0 + 1;
  int count = validRows * validCols;

  // Convert global x-range to tile column indices
  int tileColStart = x0 - tileStartX;
  int tileColEnd   = x1 - tileStartX;

  float sum = 0.0f;

  // Check if the full vertical range is inside the image
  // If yes, colSum values are correct (no excess zeros counted)
  if (outY - BLUR_SIZE >= 0 && outY + BLUR_SIZE < height) {
    // Interior pixel: all rows valid, column sums are exact
    for (int c = tileColStart; c <= tileColEnd; c++) {
      sum += colSum[threadIdx.y][c];
    }
  } else {
    // Border pixel: some rows were out-of-bounds (loaded as 0).
    // Recompute partial column sums for only the valid row range.
    int validRowStart = y0 - tileStartY;
    int validRowEnd   = y1 - tileStartY;

    for (int c = tileColStart; c <= tileColEnd; c++) {
      float vsum = 0.0f;
      for (int r = validRowStart; r <= validRowEnd; r++) {
        vsum += tile[r][c];
      }
      sum += vsum;
    }
  }

  // Write the averaged result
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

  timespec timer = tic();

  ////////////////////////////////////////////////
  //@@ INSERT AND UPDATE YOUR CODE HERE

  int numBlocksX = (imageWidth + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int numBlocksY = (imageHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(numBlocksX, numBlocksY);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  size_t numBytes = imageWidth * imageHeight * sizeof(float);

  // Optimization: use pinned (page-locked) host memory for faster DMA transfers
  float *pinnedInput;
  float *pinnedOutput;
  wbCheck(cudaMallocHost((void **)&pinnedInput, numBytes));
  wbCheck(cudaMallocHost((void **)&pinnedOutput, numBytes));

  // Copy input data into pinned memory
  memcpy(pinnedInput, hostInputImageData, numBytes);

  // Allocate device memory
  wbCheck(cudaMalloc((void **)&deviceInputImageData, numBytes));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, numBytes));

  // Transfer from pinned host memory to device (faster than pageable memory)
  wbCheck(cudaMemcpy(deviceInputImageData, pinnedInput,
                     numBytes, cudaMemcpyHostToDevice));

  // Run the optimized blur kernel 10 times for timing stability
  for (int i = 0; i < 10; i++) {
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData,
                                      deviceInputImageData,
                                      imageWidth,
                                      imageHeight);
    wbCheck(cudaGetLastError());
  }

  wbCheck(cudaDeviceSynchronize());

  // Transfer from device to pinned host memory (faster DMA path)
  wbCheck(cudaMemcpy(pinnedOutput, deviceOutputImageData,
                     numBytes, cudaMemcpyDeviceToHost));

  // Copy from pinned memory back to the output image buffer
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

  // Free all memory
  cudaFreeHost(pinnedInput);
  cudaFreeHost(pinnedOutput);
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);
  wbImage_delete(goldImage);

  return 0;
}
