#ifndef SPARSEMATRIX_GPU_H
#define SPARSEMATRIX_GPU_H

#include <cuda.h>
#include "SparseMatrix.h"

class SparseMatrix_GPU {
    public: 
        /* Copy Constructor */
        __host__ __device__ SparseMatrix_GPU(const SparseMatrix & s);
        __host__ __device__ ~SparseMatrix_GPU();


        int numberOfRows, numberOfColumns, size;
        // These are for the current matrix
        int * old_values_ref;
        // These are for the next matrix
        int * new_values_dev;
};

#endif
