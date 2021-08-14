#ifndef CSR_GPU_H
#define CSR_GPU_H

#include "SparseMatrix_GPU.cuh"
#include "COO_GPU.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


class CSR_GPU : public SparseMatrix_GPU {
    public:
        // Copy constructor for allocating graph object
        __host__ __device__ CSR(const CSR & c);

        // These are the current CSR
        thrust::device_vector<int> * old_column_indices_ref, * old_row_offsets_ref;
        int vertexCount;

        // These are for the next CSR
        thrust::host_vector<int> new_column_indices;
        thrust::host_vector<int> new_row_offsets;

        thrust::device_vector<int> new_column_indices_dev;
        thrust::device_vector<int> new_row_offsets_dev;

};

#endif
