#ifdef FPT_CUDA

#include "CSR_GPU.cuh"

__host__ __device__ CSR_GPU::CSR_GPU(int vertexCount):vertexCount(vertexCount),
SparseMatrix_GPU(vertexCount){

}

/* First graph */
__host__ __device__ CSR_GPU::CSR_GPU(int vertexCount, 
                                        int size,
                                        int numberOfRows,
                                        int * old_row_offsets_dev,
                                        int * old_columns_dev,
                                        int * old_values_dev,
                                        int * new_row_offsets_dev,
                                        int * new_columns_dev,
                                        int * new_values_dev):
vertexCount(vertexCount),
SparseMatrix_GPU(vertexCount, size, numberOfRows, old_values_dev, new_values_dev)
{
    this->old_row_offsets_ref = new array_container(old_columns_dev, 0, numberOfRows+1);
    this->old_column_indices_ref = new array_container(old_columns_dev, 0, size);
    this->new_row_offsets_dev = new array_container(new_row_offsets_dev, 0, numberOfRows+1);
    this->new_column_indices_dev = new array_container(new_columns_dev, 0, size);

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

