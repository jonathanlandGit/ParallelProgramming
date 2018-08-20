
#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include "imageFile.h"
#include <time.h>
#include <string.h>
#include <math.h>

//macro to check return value of the cuda runtime call and exits 
//if call failed
#define cudaCheck(value) {                                                                                          \
                cudaError_t _m_cudaStat = value;                                                                                \
                if (_m_cudaStat != cudaSuccess) {                                                                               \
                        fprintf(stderr, "Error %s at line %d in file %s\n",                                     \
                                        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);           \
                                        exit(1);                                                                                                        \
                } }

__global__ void edgeDetect(unsigned char* device_input_data, unsigned char* device_output_data, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //check bounds
    if (x < 1 || x > width - 1 || y > height - 1 || y < 1)
        return;

    //for horizontal lines
    const int fmat_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    // for vertical lines
    const int fmat_y[3][3]  = {
        {-1, -2, -1},
        {0,   0,  0},
        {1,   2,  1}
    };

    double G_x = 0;
        double G_y = 0;
        int G;
//go through rows and cols
        for (int i = y - 3 / 2; i < y + 3 - 3 / 2; i++) {
                for (int j = x - 3 / 2; j < x + 3 - 3 / 2; j++) {
                        G_x += (double)(fmat_x[i - y + 3 / 2][x - j + 3 / 2] * device_input_data[i * width + j]);
                        G_y += (double)(fmat_y[i - y + 3 / 2][x - j + 3 / 2] * device_input_data[i * width + j]);
                }
        }

        G = sqrt(G_x * G_x + G_y * G_y);

    if (G < 0)
        G = 0;
    if (G > 255)
        G = 255;

    device_output_data[y * width + x] = G;
}


int main(int argc, char* argv[])
{

        Photo* inputPhoto;
        int width;
        int height;

  // int blockSize;
  // int minGridSize;

    char* infile;
    char* outfile;

        unsigned char *buff;
    unsigned char *buffer_out;

        infile = argv[1];
    outfile = argv[2];

    //readImage passed in and place in var, then get h and w of image
    inputPhoto = readImage(infile);
    width = getImageWidth(inputPhoto);
    height = getImageHeight(inputPhoto);

        //we need to keep track of image size
        int ImageSize = width*height*sizeof(unsigned char);
        cudaCheck(cudaMalloc(&buff, ImageSize));
         cudaCheck(cudaMalloc(&buffer_out,ImageSize));
        cudaCheck(cudaMemcpy(buff, inputPhoto->data, ImageSize, cudaMemcpyHostToDevice));

        //set grid and block sizes
        //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*) anyMethod,0, width * height);

        //int gridSize = (width * height + blockSize - 1);
        //decided to try 8, 8 in this format. I was just playing around with this and it changed the picture so I 
//went with this.

        dim3 threadsPerBlock(8,8);
        dim3 numBlocks((width)/8,(height)/8);

        //timings
        cudaEvent_t start, end;
         float elapsedTime;

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));

        //need to record an event
    cudaCheck(cudaEventRecord(start));

        edgeDetect <<<threadsPerBlock, numBlocks>>>(buff, buffer_out, height, width);
        cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaEventRecord(end));
    //compute elapsed time

        cudaCheck(cudaEventSynchronize(end));
        cudaCheck(cudaEventElapsedTime(&elapsedTime, start, end));
//      cudaCheck(cudaMemcpy(inputPhoto->data, buffer_out,ImageSize, cudaMemcpyDeviceToHost));
   cudaCheck(cudaMemcpy(inputPhoto->data, buffer_out,ImageSize, cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(buffer_out));
    cudaCheck(cudaFree(buff));
// cudaCheck(cudaFree(buf));

    writeImage(inputPhoto, outfile);
    printf("%s Edge detection took %f sec\n",infile,(elapsedTime/1000.0));

return 0;
}
