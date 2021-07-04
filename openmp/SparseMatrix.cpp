#include "SparseMatrix.h"

/* Will be useful for parallel builds */
SparseMatrix::SparseMatrix(int size, int numberOfRows, int numberOfColumns):
size(size), 
numberOfRows(numberOfRows), 
numberOfColumns(numberOfColumns),
old_values_ref(new_values)
{};
/* Will use AddEdge to build, thus size is dynamic */
SparseMatrix::SparseMatrix(int numberOfRows, int numberOfColumns):
numberOfRows(numberOfRows), 
numberOfColumns(numberOfColumns),
old_values_ref(new_values)
{};

/* Only used for creating first CSR from a COO */
SparseMatrix::SparseMatrix(SparseMatrix & s):
size(s.size), 
numberOfRows(s.numberOfRows), 
numberOfColumns(s.numberOfColumns),
// A Reference for read-only purposes
old_values_ref(s.new_values){
    std::cout << "setting size, numRows, numCols, oldValRef, and newVals" << std::endl;
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

