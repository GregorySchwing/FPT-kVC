#include "SparseMatrix_GPU.cuh"
#include <memory>

/* Copy constr */
SparseMatrix_GPU::SparseMatrix_GPU(const SparseMatrix & s):
size(s.size), 
numberOfRows(s.numberOfRows), 
numberOfColumns(s.numberOfColumns){
//    std::cout << "Setting size, numRows, numCols - Reserving new_vals" << std::endl;
    // A copy for writing purposes
    new_values.reserve(s.size);
}

