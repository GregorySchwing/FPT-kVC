#ifndef COO_CUH
#define COO_CUH
//#include "SparseMatrix.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib> /* rand */
#include <vector>
#include <string>
#include <iostream>



__global__
void add(int n, float *x, float *y);

//__host__
void initialAllocation(int * column_indices, int * row_indices, int * values, int size);


void wrapperFunction(int * column_indices, int * row_indices, int * values, int size);
#endif