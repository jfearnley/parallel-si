#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "gpu_util.h"

using namespace std;

void gpu_assert(cudaError code)
{
    if (code != cudaSuccess)
    {
        cout << "Cuda error: " << cudaGetErrorString(code) << endl;
        exit(1);
    }
}

int nblocks(int n)
{
    return n / BLOCK_SIZE + (n % BLOCK_SIZE == 0 ? 0 : 1);
}

void init_zero(int** dev_ptr, int size)
{
    int bytes = size * sizeof(int);
    gpu_assert(cudaMalloc((void**)dev_ptr, bytes)); 
    gpu_assert(cudaMemset(*dev_ptr, 0, bytes));
}

// Create and initialize a device array using vector vec
void init_from_vector(int** dev_ptr, vector<int>& vec)
{
    int bytes = vec.size() * sizeof(int);
    gpu_assert(cudaMalloc((void**)dev_ptr, bytes)); 
    gpu_assert(cudaMemcpy(*dev_ptr, vec.data(), bytes, cudaMemcpyHostToDevice));
}

void print_device_array(int* arr, int len)
{
    int* h_ptr = (int*)malloc(len * sizeof(int));

    gpu_assert(cudaMemcpy(h_ptr, arr, len*sizeof(int), cudaMemcpyDeviceToHost));

    for(int i = 0; i < len; i++)
        cout << h_ptr[i] << " "; 
    cout << endl;
}
