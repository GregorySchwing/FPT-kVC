#include "CSR_GPU.cuh"

/* Copy constructor */
__host__ CSR_GPU::CSR_GPU(const CSR & c):
SparseMatrix_GPU(c)
{
    cudaMalloc(&new_row_offsets_dev, c.new_row_offsets.capacity()*sizeof(int)); 
    cudaMalloc(&new_column_indices_dev, c.new_column_indices.capacity()*sizeof(int)); 

}

__host__ __device__ CSR_GPU::~CSR_GPU(){
    cudaFree(new_row_offsets_dev);
    cudaFree(new_column_indices_dev);
}


