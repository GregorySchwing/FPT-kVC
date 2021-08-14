#ifndef SPARSEMATRIX_GPU_H
#define SPARSEMATRIX_GPU_H

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "SparseMatrix.h"

class SparseMatrix_GPU {
    public: 
        /* Copy Constructor */
        __host__ SparseMatrix_GPU(const SparseMatrix & s);

        int numberOfRows, numberOfColumns, size;
        // These are for the current matrix
        thrust::device_vector<int> * old_values_ref;
        // These are for the next matrix
        
        thrust::host_vector<int> new_values;
        thrust::device_vector<int> new_values_dev;
};

#endif
