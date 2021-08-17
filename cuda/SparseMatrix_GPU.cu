#ifdef FPT_CUDA

#include "SparseMatrix_GPU.cuh"
#include <memory>

/* Copy constr */
__host__ __device__ SparseMatrix_GPU::SparseMatrix_GPU(int vertexCount, 
                                                        int size,
                                                        int numberOfRows,
                                                        int ** old_values_dev,
                                                        int ** new_values_dev):
size(size), 
numberOfRows(numberOfRows){
    this->old_values_ref = new array_container(old_values_dev, 0, size);
    this->new_values_dev = new array_container(new_values_dev, 0, size);
}

/* Copy constr */
__host__ __device__ SparseMatrix_GPU::SparseMatrix_GPU(const SparseMatrix & s):
size(s.size), 
numberOfRows(s.numberOfRows){
//    std::cout << "Setting size, numRows, numCols - Reserving new_vals" << std::endl;
    // A copy for writing purposes
    //cudaMalloc(&new_values_dev, s.size*sizeof(int)); 
}

__host__ __device__ SparseMatrix_GPU::~SparseMatrix_GPU(){
    //cudaFree(new_values_dev);
}

#endif