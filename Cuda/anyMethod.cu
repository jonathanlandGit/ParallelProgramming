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

__global__ void anyMethod(unsigned char* buff , unsigned char* buffer_out , int w , int h)
{
  int x = blockIdx.x * blockDim.x +threadIdx.x ;
        int y = blockIdx.y * blockDim.y +threadIdx.y;
        int width = w , height = h;

        if((x>=0 && x < width) && (y>=0 && y<height))
        {
                int hx = -buff[width*(y-1) + (x-1)] + buff[width*(y-1)+(x+1)]
                         -2*buff[width*(y)+(x-1)] + 2* buff[width*(y)+(x+1)]
                         -buff[width*(y+1)+(x-1)] + buff[width*(y+1)+(x+1)];

                int vx = buff[width*(y-1)+(x-1)] +2*buff[width*(y-1)+(x+1)] +buff[width*(y-1)+(x+1)]
                         -buff[width*(y+1)+(x-1)] -2* buff[width*(y+1)+(x)] - buff[width*(y+1)+(x+1)];
                //this is the main part changed to get the sort of tie dye effect for at least 
                //the first part of the picture 
                hx = hx*4;
                vx = vx/5;

                int val = (int)sqrt((float)(hx) * (float)(hx) + (float)(vx) * (float)(vx));

                buffer_out[y * width + x] = (unsigned char) val;
        }
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

        anyMethod <<<threadsPerBlock, numBlocks>>>(buff, buffer_out, height, width);
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
    printf("%s Any method took %f sec\n",infile,(elapsedTime/1000.0));

return 0;
}
