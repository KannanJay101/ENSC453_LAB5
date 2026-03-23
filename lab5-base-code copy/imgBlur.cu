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
#ifndef C
#define C 4
#endif

__global__ void blurKernel(float *out, const float *in, int width, int height) {
    // Shared memory allocated for the tile and its horizontal halos
    // size = BLOCK_SIZE rows * (C * BLOCK_SIZE + 2 * BLUR_SIZE) columns
    __shared__ float sh_col_sums[BLOCK_SIZE][C * BLOCK_SIZE + 2 * BLUR_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int global_r = blockIdx.y * BLOCK_SIZE + ty;
    
    // Total elements to load and calculate in shared memory by this block
    int total_elements = BLOCK_SIZE * (C * BLOCK_SIZE + 2 * BLUR_SIZE);
    int tid = ty * BLOCK_SIZE + tx;

    // Phase 1: Collaboratively calculate vertical sums for all required columns
    for (int i = tid; i < total_elements; i += BLOCK_SIZE * BLOCK_SIZE) {
        int local_r = i / (C * BLOCK_SIZE + 2 * BLUR_SIZE);
        int local_c = i % (C * BLOCK_SIZE + 2 * BLUR_SIZE);

        int current_global_r = blockIdx.y * BLOCK_SIZE + local_r;
        int current_global_c = blockIdx.x * (C * BLOCK_SIZE) - BLUR_SIZE + local_c;

        float v_sum = 0.0f;
        // Check if the current column is within the image bounds
        // If it's outside, it will safely store 0.0f.
        if (current_global_c >= 0 && current_global_c < width) {
            int r_min = max(0, current_global_r - BLUR_SIZE);
            int r_max = min(height - 1, current_global_r + BLUR_SIZE);
            
            for (int r = r_min; r <= r_max; ++r) {
                v_sum += in[r * width + current_global_c];
            }
        }
        sh_col_sums[local_r][local_c] = v_sum;
    }

    // Wait for all threads to finish computing their subset of the vertical sums
    __syncthreads();

    // Phase 2: Compute the horizontal sums and write to output
    if (global_r < height) {
        // Each thread calculates C output pixels horizontally!
        for (int iter = 0; iter < C; ++iter) {
            int local_out_c = iter * BLOCK_SIZE + tx;
            int global_c = blockIdx.x * (C * BLOCK_SIZE) + local_out_c;

            if (global_c < width) {
                float h_sum = 0.0f;
                // Add the pre-calculated vertical sums horizontally
                for (int dy = 0; dy <= 2 * BLUR_SIZE; ++dy) {
                    h_sum += sh_col_sums[ty][local_out_c + dy];
                }

                // Compute exact number of valid pixels by calculating bounded rectangle dimension
                int r_min = max(0, global_r - BLUR_SIZE);
                int r_max = min(height - 1, global_r + BLUR_SIZE);
                int c_min = max(0, global_c - BLUR_SIZE);
                int c_max = min(width - 1, global_c + BLUR_SIZE);
                int numPixels = (r_max - r_min + 1) * (c_max - c_min + 1);

                out[global_r * width + global_c] = h_sum / (float)numPixels;
            }
        }
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

  // Pinned memory for faster CPU-GPU DMA transfers
  float *pinnedInput, *pinnedOutput;
  wbCheck(cudaMallocHost((void **)&pinnedInput, numBytes));
  wbCheck(cudaMallocHost((void **)&pinnedOutput, numBytes));
  memcpy(pinnedInput, hostInputImageData, numBytes);

  wbCheck(cudaMalloc((void **)&deviceInputImageData, numBytes));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, numBytes));

  int numBlocksX = (imageWidth + C * BLOCK_SIZE - 1) / (C * BLOCK_SIZE);
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

  toc(&timer, "GPU execution time (Data Transfer + Shared Mem Optimization) in seconds");

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

  cudaFreeHost(pinnedInput);
  cudaFreeHost(pinnedOutput);
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);
  wbImage_delete(goldImage);

  return 0;
}
