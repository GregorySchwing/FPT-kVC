#include "SparseMatrix.h"
#include <memory>

/* Will be useful for parallel builds */
SparseMatrix::SparseMatrix(int size, int numberOfRows, int numberOfColumns):
size(size), 
numberOfRows(numberOfRows), 
numberOfColumns(numberOfColumns),
old_values_ref(&new_values)
{};

SparseMatrix::SparseMatrix(int numberOfRows, int numberOfColumns):
numberOfRows(numberOfRows), 
numberOfColumns(numberOfColumns),
old_values_ref(&new_values)
{};

/* Will use AddEdge to build, thus size is dynamic */
SparseMatrix::SparseMatrix():
old_values_ref(&new_values)
{};

/* Copy constr */
SparseMatrix::SparseMatrix(const SparseMatrix & s):
size(s.size), 
numberOfRows(s.numberOfRows), 
numberOfColumns(s.numberOfColumns){
    std::cout << "Setting size, numRows, numCols - Reserving new_vals" << std::endl;
    // A copy for writing purposes
    new_values.reserve(s.size);
}

// Build first graph
SparseMatrix::SparseMatrix(SparseMatrix & s):
size(s.size), 
numberOfRows(s.numberOfRows), 
numberOfColumns(s.numberOfColumns),
// The new vals are reserved but not initiated
old_values_ref(&s.new_values){
    std::cout << "Setting size, numRows, numCols, old(RowOff/Col/Val)Refs;" << std::endl;
    std::cout << "Setting newVals" << std::endl;
    // A copy for writing purposes
    new_values = s.new_values;
}
/* SPM by reference 
SparseMatrix::SparseMatrix(int numberOfRows, std::vector<int> & values_ref_arg):
numberOfRows(s.numberOfRows), 
numberOfColumns(s.numberOfColumns),
old_values_ref(values_ref_arg){
    // New Values are always a write-able copy of old_values_ref
    // A chance for optimization is to set this by reference in
    // Cases where there is only 1 branch
    new_values.resize(values_ref_arg.size(), 1);
}*/


int SparseMatrix::GetNumberOfRows(){
    return numberOfRows;
}

void SparseMatrix::SetNumberOfRows(int numberOfRows_arg){
    numberOfRows = numberOfRows_arg;
}


