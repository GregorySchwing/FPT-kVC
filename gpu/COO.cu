
#include <iostream>
#include <math.h>
#include <time.h> 
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "COO.cuh"

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
  {
    y[i] = x[i] + y[i];
  }
}