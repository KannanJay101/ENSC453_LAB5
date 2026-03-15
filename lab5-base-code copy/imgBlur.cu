#include "libwb/wb.h"
#include "my_timer.h"

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
__global__ void blurKernel(float *out, float *in, int width, int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x; // calculate the x index of the current thread
  int y = blockIdx.y * blockDim.y + threadIdx.y; // calculate the y index of the current thread

  if (x < width && y < height) { // check if the current thread is within the bounds of the image
    float sum = 0; // initialize the sum of the pixels to 0
    int count = 0; // initialize the count of the pixels to 0
    for (int i = -BLUR_SIZE / 2; i <= BLUR_SIZE / 2; i++) { // iterate over the blur size
      for (int j = -BLUR_SIZE / 2; j <= BLUR_SIZE / 2; j++) { // iterate over the blur size
        int nx = x + i;
        int ny = y + j; // calculate the y index of the current pixel
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          sum += in[ny * width + nx]; // add the pixel value to the sum
          count++;
        }
      }
    }
    out[y * width + x] = sum / count;
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

  int numBlocksX = (imageWidth + BLOCK_SIZE - 1) / BLOCK_SIZE; // calculate the number of blocks in the x direction
  int numBlocksY = (imageHeight + BLOCK_SIZE - 1) / BLOCK_SIZE; // calculate the number of blocks in the y direction
  dim3 dimGrid(numBlocksX, numBlocksY); // create a grid of blocks
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // create a block of threads

  // Allocate cuda memory for device input and ouput image data
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(float)); // allocate memory for the input image data
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float)); // allocate memory for the output image data

  // Transfer data from CPU to GPU
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyHostToDevice); // copy the input image data from the host to the device

  // Call your GPU kernel 10 times
  for(int i = 0; i < 10; i++) {
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData,
                                      deviceInputImageData, imageWidth,
                                      imageHeight);
    if (i < 9) {
      float *temp = deviceInputImageData;
      deviceInputImageData = deviceOutputImageData;
      deviceOutputImageData = temp;
    }
  }

  // Transfer data from GPU to CPU
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  ///////////////////////////////////////////////////////

  // Stop and print timer
  toc(&timer, "GPU execution time (including data transfer) in seconds");

  // Check the correctness of your solution
  //wbSolution(args, outputImage);

   for(int i=0; i<imageHeight; i++){
     for(int j=0; j<imageWidth; j++){
       if(fabs(hostOutputImageData[i*imageWidth+j]-goldOutputImageData[i*imageWidth+j])/goldOutputImageData[i*imageWidth+j]>0.01){
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
