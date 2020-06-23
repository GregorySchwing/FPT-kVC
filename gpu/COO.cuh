#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


__global__
void add(int n, float *x, float *y);