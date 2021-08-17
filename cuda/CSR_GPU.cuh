#ifndef CSR_GPU_H
#define CSR_GPU_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#ifdef FPT_CUDA

#include "SparseMatrix_GPU.cuh"
#include "SparseMatrix.h"
#include "CSR.h"
#include <cuda.h>


class CSR_GPU : public SparseMatrix_GPU {
    public:
        CUDA_HOSTDEV CSR_GPU(int vertexCount);

        // Default constructor for first graph object
        CUDA_HOSTDEV CSR_GPU(int vertexCount, 
                            int size,
                            int numberOfRows,
                            int ** old_row_offsets_dev,
                            int ** old_columns_dev,
                            int ** old_values_dev,
                            int ** new_row_offsets_dev,
                            int ** new_columns_dev,
                            int ** new_values_dev);
        // Copy constructor for allocating graph object
        CUDA_HOSTDEV CSR_GPU(const CSR & c);
        CUDA_HOSTDEV ~CSR_GPU();

        

        // These are the current CSR
        array_container * old_column_indices_ref, * old_row_offsets_ref;
        int vertexCount;

        // These are for the next CSR
        array_container * new_column_indices_dev;
        array_container * new_row_offsets_dev;

};

#endif
#endif
