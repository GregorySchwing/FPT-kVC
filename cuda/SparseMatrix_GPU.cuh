#ifndef SPARSEMATRIX_GPU_H
#define SPARSEMATRIX_GPU_H

#include <cuda.h>
#include "SparseMatrix.h"


struct array_container {
    __device__ array_container(int ** globalData, long long globalIndex, int count_arg) : count(count_arg){
        data = globalData[globalIndex];
    }
    int * data;
    int count;
};

class SparseMatrix_GPU {
    public: 
        /* Copy Constructor */
        __host__ __device__ SparseMatrix_GPU(const SparseMatrix & s);
        __host__ __device__ ~SparseMatrix_GPU();


        int numberOfRows, numberOfColumns, size;
        // These are for the current matrix
        array_container * old_values_ref;
        // These are for the next matrix
        array_container * new_values_dev;
};

#endif
