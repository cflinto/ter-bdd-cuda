#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ROW_NUM 10000100
#define COLUMN_NUM 4

__global__
void request(int *tab, int *result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < ROW_NUM; i += stride)
    {
        if(result[i] > 10000)
        {
            result[i] = 1;
        }
        else
        {
            result[i] = 0;
        }
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int main(void)
{
    int *tab, *result;
    
    cudaMallocManaged(&tab, ROW_NUM*COLUMN_NUM*sizeof(int));
    cudaMallocManaged(&result, ROW_NUM*sizeof(int));
    
    srand(0);
    
    for(int column=0;column<COLUMN_NUM-1;++column)
    {
        for(int row=0;row<ROW_NUM;++row)
        {
            tab[ROW_NUM*column+row] = rand()%1000000;
        }
    }
    
    //t1 = myCPUTimer();
    
    request<<<(ROW_NUM+255)/256, 256>>>(tab, result);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    //t2 = myCPUTimer();
    
    int total = 0;
    for(int row=0;row<ROW_NUM;++row)
    {
        if(result[row])
        {
            ++total;
        }
    }
    std::cout << "Total : " << total << std::endl;
    
    cudaFree(tab);
    cudaFree(result);
      
    return 0;
}