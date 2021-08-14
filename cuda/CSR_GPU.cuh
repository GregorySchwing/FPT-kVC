#ifndef CSR_GPU_H
#define CSR_GPU_H

#include "SparseMatrix_GPU.cuh"
#include "SparseMatrix.h"
#include "CSR.h"
#include <cuda.h>


class CSR_GPU : public SparseMatrix_GPU {
    public:
        // Copy constructor for allocating graph object
        __host__ __device__ CSR_GPU(const CSR & c);
        __host__ __device__ ~CSR_GPU();

        

        // These are the current CSR
        array_container * old_column_indices_ref, * old_row_offsets_ref;
        int vertexCount;

        // These are for the next CSR
        array_container * new_column_indices_dev;
        array_container * new_row_offsets_dev;

};

#endif
