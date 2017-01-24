#ifndef GPU_UTIL
#define GPU_UTIL

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 128

void gpu_assert(cudaError code);
int nblocks(int n);
void init_zero(int** dev_ptr, int size);
void init_from_vector(int** dev_ptr, std::vector<int>& vec);
void print_device_array(int* arr, int len);

#endif
