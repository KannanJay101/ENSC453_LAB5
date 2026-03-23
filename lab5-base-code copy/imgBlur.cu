#include "libwb/wb.h"
#include "my_timer.h"
#include <cmath>

#define wbCheck(stmt)							\
  do {									\
    cudaError_t err = stmt;						\
    if (err != cudaSuccess) {						\
      wbLog(ERROR, "Failed to run stmt ", #stmt);			\
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));	\
      return -1;							\
    }									\
  } while (0)

#define BLUR_SIZE 21
#define BLOCK_SIZE 16

///////////////////////////////////////////////////////
//@@ INSERT YOUR CODE HERE
#ifndef A
#define A 1
#endif

__global__ void blurKernel(float *out, const float *in, int width, int height) {
    // 1. Allocate Shared Memory: exact dimensions based on BLOCK_DIM and chosen multiplier a.
    __shared__ float sh_pixels[BLOCK_SIZE * A][BLOCK_SIZE * A];
    __shared__ float base_sum;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int global_c = blockIdx.x * (BLOCK_SIZE * A) + tx;
    int global_r = blockIdx.y * (BLOCK_SIZE * A) + ty;

    // 2. Map Threads to the Tile: standard block threads load data from global memory into shared memory.
    if (global_r < height && global_c < width) {
        sh_pixels[ty][tx] = in[global_r * width + global_c];
    } else {
        sh_pixels[ty][tx] = 0.0f;
    }
    __syncthreads();

    // Collaboratively compute the shared base sum of the inner tile
    // To minimize synchronization complexity, thread 0 sums the shared memory array.
    if (tx == 0 && ty == 0) {
        float b_sum = 0.0f;
        for (int i = 0; i < BLOCK_SIZE * A; ++i) {
            for (int j = 0; j < BLOCK_SIZE * A; ++j) {
                b_sum += sh_pixels[i][j];
            }
        }
        base_sum = b_sum;
    }
    __syncthreads();

    // 3. Compute: pull the base sum from shared memory and loop only over unique fringe pixels
    if (global_r < height && global_c < width) {
        float pixelValue = base_sum; // Pull base sum

        // Determine the bounds of the base tile in global coordinates
        int tile_r_min = blockIdx.y * (BLOCK_SIZE * A);
        int tile_r_max = min(height - 1, tile_r_min + BLOCK_SIZE * A - 1);
        int tile_c_min = blockIdx.x * (BLOCK_SIZE * A);
        int tile_c_max = min(width - 1, tile_c_min + BLOCK_SIZE * A - 1);

        int r_min = max(0, global_r - BLUR_SIZE);
        int r_max = min(height - 1, global_r + BLUR_SIZE);
        int c_min = max(0, global_c - BLUR_SIZE);
        int c_max = min(width - 1, global_c + BLUR_SIZE);
        
        // Loop over the unique fringe pixels
        for (int r = r_min; r <= r_max; ++r) {
            for (int c = c_min; c <= c_max; ++c) {
                // Determine if this pixel is inside the precomputed base_sum tile
                bool in_shared_tile = (r >= tile_r_min && r <= tile_r_max && 
                                       c >= tile_c_min && c <= tile_c_max);
                
                // Add only the fringe pixels from global memory
                if (!in_shared_tile) {
                    pixelValue += in[r * width + c];
                }
            }
        }

        int numPixels = (r_max - r_min + 1) * (c_max - c_min + 1);
        out[global_r * width + global_c] = pixelValue / (float)numPixels;
    }
}
///////////////////////////////////////////////////////

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

  wbCheck(cudaMalloc((void **)&deviceInputImageData, numBytes));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, numBytes));

  int numBlocksX = (imageWidth + A * BLOCK_SIZE - 1) / (A * BLOCK_SIZE);
  int numBlocksY = (imageHeight + A * BLOCK_SIZE - 1) / (A * BLOCK_SIZE);
  dim3 dimGrid(numBlocksX, numBlocksY);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  timespec timer = tic();

  ////////////////////////////////////////////////
  // Standard pageable memory transfer
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData,
                     numBytes, cudaMemcpyHostToDevice));

  // 10 kernel repeats — same as baseline
  for (int i = 0; i < 10; i++) {
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData,
                                      deviceInputImageData,
                                      imageWidth, imageHeight);
    wbCheck(cudaGetLastError());
  }

  wbCheck(cudaDeviceSynchronize());

  // Standard pageable memory transfer back
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData,
                     numBytes, cudaMemcpyDeviceToHost));

  ///////////////////////////////////////////////////////

  toc(&timer, "GPU execution time (Shared Memory & Comp Reuse Optimization) in seconds");

  // Correctness check against golden output
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
