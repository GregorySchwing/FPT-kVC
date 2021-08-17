#ifndef SPARSEMATRIX_GPU_H
#define SPARSEMATRIX_GPU_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif

#ifdef FPT_CUDA

#include <cuda.h>
#include "SparseMatrix.h"


struct array_container {
    CUDA_DEV array_container(int ** globalData, long long globalIndex, int count_arg) : count(count_arg){
        data = globalData[globalIndex];
    }
    int * data;
    int count;
};

class SparseMatrix_GPU {
    public: 
        /* First Graph Constructor */
        CUDA_HOSTDEV SparseMatrix_GPU(int vertexCount, 
                                    int size,
                                    int numberOfRows);

        /* Copy Constructor */
        CUDA_HOSTDEV SparseMatrix_GPU(const SparseMatrix & s);
        CUDA_HOSTDEV ~SparseMatrix_GPU();


        int numberOfRows, size;
        // These are for the current matrix
        array_container * old_values_ref;
        // These are for the next matrix
        array_container * new_values_dev;
};

#endif
#endif
