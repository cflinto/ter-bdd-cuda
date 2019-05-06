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
        if(tab[i] > 1000 && tab[i] < 100000
        && tab[i+ROW_NUM] > 1000 && tab[i+ROW_NUM] < 100000
        && tab[i+ROW_NUM*2] > 1000 && tab[i+ROW_NUM*2] < 100000)
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
    int *tab, *result; // GPU
    int *tabCPU, *resultCPU; // CPU
    
    tabCPU = new int[ROW_NUM*COLUMN_NUM];
    resultCPU = new int[ROW_NUM];
    
    cudaMalloc(&tab, ROW_NUM*COLUMN_NUM*sizeof(int));
    cudaMalloc(&result, ROW_NUM*sizeof(int));
    
    srand(0);
    
    for(int column=0;column<COLUMN_NUM-1;++column)
    {
        for(int row=0;row<ROW_NUM;++row)
        {
            tabCPU[ROW_NUM*column+row] = rand()%1000000;
        }
    }
    
    cudaMemcpy(tab, tabCPU, ROW_NUM*COLUMN_NUM*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(result, resultCPU, ROW_NUM*sizeof(int), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    request<<<(ROW_NUM+255)/256, 256>>>(tab, result);
    
    cudaEventRecord(stop);
    
    cudaMemcpy(resultCPU, result, ROW_NUM*sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    /*
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize());
    */
    
    
    int total = 0;
    for(int row=0;row<ROW_NUM;++row)
    {
        if(result[row])
        {
            ++total;
        }
    }
    std::cout << "Total : " << total << std::endl;
    
    
    std::cout << milliseconds;
    
    cudaFree(tab);
    cudaFree(result);
    
    delete[] resultCPU;
    delete[] tabCPU;
      
    return 0;
}