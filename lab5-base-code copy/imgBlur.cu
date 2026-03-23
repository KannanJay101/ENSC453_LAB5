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
  // 1. Thread Mapping: Calculate the global (x, y) coordinate for this thread
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // 2. Bounds Checking: Ensure the thread maps to a valid pixel inside the image
  if (col < width && row < height) {
    float pixelValue = 0.0f; 
    int numPixels = 0;       

    // 3. The Blur Window: Iterate over the surrounding pixels
    // BLUR_SIZE is the radius (21). The window is (2*BLUR_SIZE+1) x (2*BLUR_SIZE+1)
    for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
      for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
        
        int curRow = row + blurRow;
        int curCol = col + blurCol;

        // 4. Inner Bounds Check: Ensure the neighboring pixel isn't off the edge of the image
        if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
          
          // 5. Accumulate: Flatten the 2D index to 1D to read from the global 'in' array
          pixelValue += in[curRow * width + curCol];
          numPixels++; 
        }
      }
    }

    // 6. Average and Output: Write the final blurred value to the global 'out' array
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

  // ---------------------------------------------------------
  // OPTIMIZATION: Allocate Pinned (Page-Locked) Memory
  // ---------------------------------------------------------
  float *pinnedInput, *pinnedOutput;
  wbCheck(cudaMallocHost((void **)&pinnedInput, numBytes));
  wbCheck(cudaMallocHost((void **)&pinnedOutput, numBytes));
  
  // Copy image data from standard host memory to pinned memory
  memcpy(pinnedInput, hostInputImageData, numBytes);

  wbCheck(cudaMalloc((void **)&deviceInputImageData, numBytes));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, numBytes));

  int numBlocksX = (imageWidth + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int numBlocksY = (imageHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(numBlocksX, numBlocksY);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  timespec timer = tic();

  ////////////////////////////////////////////////
  // Transfer from PINNED memory to device (Faster DMA transfer)
  wbCheck(cudaMemcpy(deviceInputImageData, pinnedInput,
                     numBytes, cudaMemcpyHostToDevice));

  // Run the same blur kernel 10 times for timing stability.
  for (int i = 0; i < 10; i++) {
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData,
                                      deviceInputImageData,
                                      imageWidth,
                                      imageHeight);
    wbCheck(cudaGetLastError());
  }

  wbCheck(cudaDeviceSynchronize());

  // Transfer from device back to PINNED memory
  wbCheck(cudaMemcpy(pinnedOutput, deviceOutputImageData,
                     numBytes, cudaMemcpyDeviceToHost));
                     
  // Copy back from pinned memory to standard host memory so wbImage can use it
  memcpy(hostOutputImageData, pinnedOutput, numBytes);
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

  // Free pinned memory
  cudaFreeHost(pinnedInput);
  cudaFreeHost(pinnedOutput);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);
  wbImage_delete(goldImage);

  return 0;
}