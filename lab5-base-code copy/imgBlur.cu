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

///////////////////////////////////////////////////////
//@@ INSERT YOUR CODE HERE
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16
#define L_PARAM 3

#define OUT_WIDTH  (BLOCK_SIZE_X * L_PARAM)          // 96 output columns
#define OUT_HEIGHT BLOCK_SIZE_Y                      // 16 output rows

#define TILE_WIDTH  (OUT_WIDTH  + 2 * BLUR_SIZE)     // 96 + 42 = 138
#define TILE_HEIGHT (OUT_HEIGHT + 2 * BLUR_SIZE)     // 16 + 42 = 58

__global__ void blurKernel(float *out, float *in, int width, int height) {

  __shared__ float tile[TILE_HEIGHT][TILE_WIDTH];
  __shared__ float colSum[OUT_HEIGHT][TILE_WIDTH];

  int outBlockX = blockIdx.x * OUT_WIDTH;
  int outBlockY = blockIdx.y * OUT_HEIGHT;

  int tileStartX = outBlockX - BLUR_SIZE;
  int tileStartY = outBlockY - BLUR_SIZE;

  for (int ty = threadIdx.y; ty < TILE_HEIGHT; ty += blockDim.y) {
    for (int tx = threadIdx.x; tx < TILE_WIDTH; tx += blockDim.x) {
      int gx = tileStartX + tx;
      int gy = tileStartY + ty;

      if (gx >= 0 && gx < width && gy >= 0 && gy < height)
        tile[ty][tx] = in[gy * width + gx];
      else
        tile[ty][tx] = 0.0f; 
    }
  }

  __syncthreads();

  for (int c = threadIdx.x; c < TILE_WIDTH; c += blockDim.x) {
    float vsum = 0.0f;
    
    #pragma unroll
    for (int dy = 0; dy <= 2 * BLUR_SIZE; dy++) {
      vsum += tile[threadIdx.y + dy][c];
    }
    colSum[threadIdx.y][c] = vsum;
  }

  __syncthreads();

  int outY = outBlockY + threadIdx.y;
  if (outY >= height) return;

  int y0 = max(0, outY - BLUR_SIZE);
  int y1 = min(height - 1, outY + BLUR_SIZE);
  int validRows = y1 - y0 + 1;

  for (int p = 0; p < L_PARAM; p++) {
    int outX = outBlockX + threadIdx.x + p * blockDim.x;
    if (outX >= width) continue;

    int x0 = max(0, outX - BLUR_SIZE);
    int x1 = min(width - 1, outX + BLUR_SIZE);
    int validCols = x1 - x0 + 1;
    int count = validRows * validCols;

    int tileColStart = x0 - tileStartX;
    int tileColEnd   = x1 - tileStartX;
    
    float sum = 0.0f;
    for (int c = tileColStart; c <= tileColEnd; c++) {
      sum += colSum[threadIdx.y][c];
    }

    out[outY * width + outX] = sum / (float)count;
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

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage = wbImport(inputImageFile);

  char *goldImageFile = argv[2];
  goldImage = wbImport(goldImageFile);

  // The input image is in grayscale, so the number of channels is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  // Get host input and output image data
  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  goldOutputImageData = wbImage_getData(goldImage);

  // Start timer
  timespec timer = tic();

  ////////////////////////////////////////////////
  //@@ INSERT AND UPDATE YOUR CODE HERE
  int numBlocksX = (imageWidth + OUT_WIDTH - 1) / OUT_WIDTH;
  int numBlocksY = (imageHeight + OUT_HEIGHT - 1) / OUT_HEIGHT;
  dim3 dimGrid(numBlocksX, numBlocksY);
  dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

  size_t numBytes = imageWidth * imageHeight * sizeof(float);

  float *pinnedInput;
  float *pinnedOutput;

  wbCheck(cudaMallocHost((void **)&pinnedInput, numBytes));
  wbCheck(cudaMallocHost((void **)&pinnedOutput, numBytes));
  wbCheck(cudaMalloc((void **)&deviceInputImageData, numBytes));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, numBytes));

  memcpy(pinnedInput, hostInputImageData, numBytes);

  wbCheck(cudaFree(0));


  wbCheck(cudaMemcpy(deviceInputImageData, pinnedInput,
                     numBytes, cudaMemcpyHostToDevice));

  for (int i = 0; i < 10; i++) {
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData,
                                      deviceInputImageData,
                                      imageWidth, imageHeight);
    wbCheck(cudaGetLastError());
  }

  wbCheck(cudaDeviceSynchronize());

  // Fast DMA transfer from device to pinned host memory
  wbCheck(cudaMemcpy(pinnedOutput, deviceOutputImageData,numBytes, cudaMemcpyDeviceToHost));

  memcpy(hostOutputImageData, pinnedOutput, numBytes);  

  ///////////////////////////////////////////////////////
  // Stop and print timer
  toc(&timer, "GPU execution time (including data transfer) in seconds");

  // Check the correctness of your solution
  //wbSolution(args, outputImage);

  for(int i=0; i<imageHeight; i++){
    for(int j=0; j<imageWidth; j++){
      if(abs(hostOutputImageData[i*imageWidth+j]-goldOutputImageData[i*imageWidth+j])/goldOutputImageData[i*imageWidth+j]>0.01){
        printf("Incorrect output image at pixel (%d, %d): goldOutputImage = %f, hostOutputImage = %f\n", i, j, goldOutputImageData[i*imageWidth+j],hostOutputImageData[i*imageWidth+j]);
  return -1;
      }
    }
  }
  printf("Correct output image!\n");

 cudaFree(deviceInputImageData);
 cudaFree(deviceOutputImageData);

 wbImage_delete(outputImage);
 wbImage_delete(inputImage);

 return 0;
}