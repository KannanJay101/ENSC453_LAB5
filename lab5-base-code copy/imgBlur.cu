#include "libwb/wb.h"
#include "my_timer.h"
#include <cmath>

#define wbCheck(stmt)             \
  do {                  \
    cudaError_t err = stmt;           \
    if (err != cudaSuccess) {           \
      wbLog(ERROR, "Failed to run stmt ", #stmt);     \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));  \
      return -1;              \
    }                 \
  } while (0)

#define BLUR_SIZE 21
#define BLOCK_SIZE 16

///////////////////////////////////////////////////////
//@@ INSERT YOUR CODE HERE
__global__ void blurKernel(float *out, const float *in, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // 1. Declare Shared Memory for the "Red Region" core sum
  __shared__ float shared_core_sum;
  __shared__ int shared_core_count;

  // Initialize shared memory
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    shared_core_sum = 0.0f;
    shared_core_count = 0;
  }
  __syncthreads();

  // Define the core "Red Region" bounds based on the integer parameter 'a'
  // For BLUR_SIZE=21 and BLOCK_DIM=16, a=1 is the max size that fits in all windows.
  int a = 1;
  int core_width = a * blockDim.x;
  int core_height = a * blockDim.y;
  
  int core_start_x = blockIdx.x * blockDim.x;
  int core_start_y = blockIdx.y * blockDim.y;
  int core_end_x = core_start_x + core_width - 1;
  int core_end_y = core_start_y + core_height - 1;

  // 2. Collaboratively compute the sum of the common core region
  // Every thread adds its own pixel to the shared variable using fast atomics
  if (col <= core_end_x && row <= core_end_y && col < width && row < height) {
    atomicAdd(&shared_core_sum, in[row * width + col]);
    atomicAdd(&shared_core_count, 1);
  }
  __syncthreads();

  // 3. Compute the final blur for each thread, reusing the shared core sum
  if (col < width && row < height) {
    float pixelValue = shared_core_sum; 
    int numPixels = shared_core_count;       

    for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
      for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
        
        int curRow = row + blurRow;
        int curCol = col + blurCol;

        // SKIP the "red region" because it is already accumulated in pixelValue
        if (curRow >= core_start_y && curRow <= core_end_y && 
            curCol >= core_start_x && curCol <= core_end_x) {
          continue; 
        }

        // Add only the remaining "fringe" pixels
        if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
          pixelValue += in[curRow * width + curCol];
          numPixels++; 
        }
      }
    }

    out[row * width + col] = pixelValue / (float)numPixels;
  }
}
//////////////////////////////////////////////////////

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

  int numBlocksX = (imageWidth + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int numBlocksY = (imageHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(numBlocksX, numBlocksY);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  timespec timer = tic();

  ////////////////////////////////////////////////
  // Standard, unoptimized pageable memory transfer
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData,
                     numBytes, cudaMemcpyHostToDevice));

  // Run the blur kernel 10 times for timing stability
  for (int i = 0; i < 10; i++) {
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData,
                                      deviceInputImageData,
                                      imageWidth,
                                      imageHeight);
    wbCheck(cudaGetLastError());
  }

  wbCheck(cudaDeviceSynchronize());

  // Standard, unoptimized pageable memory transfer back
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData,
                     numBytes, cudaMemcpyDeviceToHost));
  ///////////////////////////////////////////////////////

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