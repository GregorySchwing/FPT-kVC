#include "SparseMatrix_GPU.cuh"
#include <memory>

/* Copy constr */
__host__ SparseMatrix_GPU::SparseMatrix_GPU(const SparseMatrix & s):
size(s.size), 
numberOfRows(s.numberOfRows), 
numberOfColumns(s.numberOfColumns){
//    std::cout << "Setting size, numRows, numCols - Reserving new_vals" << std::endl;
    // A copy for writing purposes
    new_values_dev.reserve(s.size);
}

__host__ SparseMatrix_GPU::~SparseMatrix_GPU(){
    new_values_dev.clear();
    new_values_dev.shrink_to_fit();
}

