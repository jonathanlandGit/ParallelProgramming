#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "ppmFile.h"

#define max(a, b) ({ \
    __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; \
})

#define min(a, b) ({ \
    __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; \
})

#define CUDA_CHECK_RETURN(value) 
{											    \
		cudaError_t _m_cudaStat = value;	
											\
		if (_m_cudaStat != cudaSuccess) 
		{										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
					exit(1);													\
		} 
}

#define X_Y_TO_INDEX(x, y, width) (((y) * (width) + (x)) * 3)


__global__ void blur(int width, int height, int r, unsigned char* input, unsigned char* output)
{
    int startRow = blockIdx.x * (height / gridDim.x) + min(blockIdx.x, (height % gridDim.x));

    int endRow = (blockIdx.x + 1) * (height / gridDim.x) + min(blockIdx.x + 1, (height % gridDim.x));

    int pixelsPerSlice = width * (endRow - startRow + 1);

    int blockStartPixel = startRow * width;

    int startPixel = threadIdx.x * (pixelsPerSlice / blockDim.x) + min(threadIdx.x, (pixelsPerSlice % blockDim.x)) + blockStartPixel;

    int endPixel = ((threadIdx.x + 1) * (pixelsPerSlice / blockDim.x) + min(threadIdx.x + 1, (pixelsPerSlice % blockDim.x))) + blockStartPixel;

    for(int k = startPixel; k < endPixel; k++)
    {
            int x = k % width;
            int y = k / width;

            int minX = max((x-r),0);
            int maxX = min((x+r), width-1);

            int minY = max((y-r),0);
            int maxY = min((y+r),height-1);

            int red = 0;
            int green = 0;
            int blue = 0;

            int count = 0;

            for (int i = minX; i<=maxX;i++)
            {
                for(int j = minY; j<=maxY;j++)
                {
                    red = red + input[X_Y_TO_INDEX(i,j,width)+0];
                    green = green + input[X_Y_TO_INDEX(i,j,width)+1];
                    blue = blue + input[X_Y_TO_INDEX(i,j,width)+2];
                    count++;
                }
            }

            red = red/count;
            green = green/count;
            blue = blue/count;

            output[X_Y_TO_INDEX(x,y,width)+0] = red;
            output[X_Y_TO_INDEX(x,y,width)+1] = green;
            output[X_Y_TO_INDEX(x,y,width)+2] = blue;
    }
    
__global__ void edgeDetection((int iw, int ih, unsigned char *source, unsigned char *dest)
{
	//calculate location of pixels
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	//now operate only if in correct bounds
	if (x > 0; && iw - 1 && y > 0 && y < ih -1)
	{
		int gx = -source[iw*(y-1)+(x-1)] + source[iw*(y-1)+(x+1) +
				-2*source(iw*(y)+(x-1)] + 2*source[iw*(y)+(x+1) +
				-source[iw*(y+1)+(x-1)] + source[iw*(y+1)+(x+1);
		
		int gy = -source[iw*(y-1)+(x-1)] - 2*source[iw*(y-1)+(x)]
 				-source[iw*(y-1)+(x+1)] +
 				source[iw*(y+1)+(x-1) + 2*source[iw*(y+1)+(x)] +
 				source(iw*(y+1)+(x+1)]; 
 		
 		dest[iw*y+x] = (int) sqrt((float)(gx)*(float)(gx) + (float)(gy)*(float)(gy));   
 	}
}

//if needed???? to get C code running
void edgeDetectionFilter(int iw, int ih, unsigned char *source, unsigned char *dest)
{
	//allocate mem in GPU
	unsigned char *dev_source, *dev_dest;
	cudaHostGetDevicePointer(&dev_source, source, 0);
	cudaHostGetDevicePointer(&dev_dest, dest, 0);

	//run filter
	dim3 blocks(iw/16, ih/16);
	dim3 threads(16,16);
	
	//now execute kernel - I suppose I should do this in main though
	edgeDetection<<<blocks, threads>>>(iw, ih, dev_source, dev_dest);
	cudaThreadSynchronize(); 

int main(int argc, char* argv[])
{
    Image* inputImage;
    int r, width, height;
    char* infile;
    char* outfile;
    unsigned char *d_inputdata;
    unsigned char *d_outputdata;

    int blockSize;
    int minGridSize;

    if (argc < 4)
    {
        printf("Error: Not Enough Arguments\n");
        return 1;
    }

    r = atoi(argv[1]);
    infile = argv[2];
    outfile = argv[3];

    inputImage = ImageRead(infile);

    width = ImageWidth(inputImage);
    height = ImageHeight(inputImage);

    CUDA_CHECK_RETURN(cudaMalloc(&d_inputdata, width*height*3*sizeof(unsigned char)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_outputdata, width*height*3*sizeof(unsigned char)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_inputdata, inputImage->data, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice));

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*) blur, 0, width * height);

    int gridSize = (width * height + blockSize - 1) / blockSize;

    cudaEvent_t start, end;
    float elapsedTime;

    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&end));

    CUDA_CHECK_RETURN(cudaEventRecord(start));
    
    blur <<< gridSize, blockSize >>> (width, height, r, d_inputdata, d_outputdata);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaEventRecord(end));
    CUDA_CHECK_RETURN(cudaEventSynchronize(end));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, end));

    CUDA_CHECK_RETURN(cudaMemcpy(inputImage->data, d_outputdata,width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(d_outputdata));
    CUDA_CHECK_RETURN(cudaFree(d_inputdata));

    ImageWrite(inputImage,outfile);
    printf("%s blurred in %f sec\n",infile,(elapsedTime/1000.0));
    return 0;
}
