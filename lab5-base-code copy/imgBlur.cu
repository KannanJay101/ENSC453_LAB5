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
// Baseline box blur kernel (global memory, no tiling)
// Each thread computes one output pixel by averaging all valid pixels
// in a (BLUR_SIZE x BLUR_SIZE) neighbourhood centred on (x, y).
// Border pixels use only the neighbours that fall inside the image
// (clamped / partial-window averaging).
__global__ void blurKernel(float *out, float *in, int width, int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

  if (x < width && y < height) {
    float sum   = 0.0f;
    int   count = 0;

    // Half-width of the blur window
    int half = BLUR_SIZE / 2;

    for (int dy = -half; dy <= half; dy++) {      // row offset
      for (int dx = -half; dx <= half; dx++) {    // column offset
        int nx = x + dx;
        int ny = y + dy;
        // Only accumulate pixels that lie inside the image (clamp-to-border)
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          sum += in[ny * width + nx];
          count++;
        }
      }
    }

    // count is always >= 1 because the centre pixel is always valid
    out[y * width + x] = sum / (float)count;
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

   // Verify output against gold image.
   // Use a mixed absolute+relative tolerance to handle near-zero gold values
   // safely (pure relative check would divide by zero for black pixels).
   const float ABS_TOL = 1e-4f;
   const float REL_TOL = 0.01f;
   for(int i = 0; i < imageHeight; i++){
     for(int j = 0; j < imageWidth; j++){
       float gold = goldOutputImageData[i*imageWidth+j];
       float out  = hostOutputImageData [i*imageWidth+j];
       float diff = fabsf(out - gold);
       float tol  = REL_TOL * fabsf(gold) + ABS_TOL;
       if(diff > tol){
         printf("Incorrect output at pixel (row=%d, col=%d): gold=%f, output=%f, diff=%f\n",
                i, j, gold, out, diff);
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