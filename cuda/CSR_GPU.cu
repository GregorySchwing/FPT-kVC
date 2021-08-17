#ifdef FPT_CUDA

#include "CSR_GPU.cuh"


/* First graph */
__host__ __device__ CSR_GPU::CSR_GPU(int vertexCount, 
                                        int size,
                                        int numberOfRows):
vertexCount(vertexCount),
SparseMatrix_GPU(vertexCount, size, numberOfRows)
{
    //cudaMalloc(&new_row_offsets_dev, (c.vertexCount + 1)*sizeof(int)); 
    //cudaMalloc(&new_column_indices_dev, c.size*sizeof(int)); 

}

/* Copy constructor */
__host__ __device__ CSR_GPU::CSR_GPU(const CSR & c):
vertexCount(c.vertexCount),
SparseMatrix_GPU(c)
{
    //cudaMalloc(&new_row_offsets_dev, (c.vertexCount + 1)*sizeof(int)); 
    //cudaMalloc(&new_column_indices_dev, c.size*sizeof(int)); 

}

__host__ __device__ CSR_GPU::~CSR_GPU(){
    //cudaFree(new_row_offsets_dev);
    //cudaFree(new_column_indices_dev);
}

#endif

