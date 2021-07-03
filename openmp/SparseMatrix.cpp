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


SparseMatrix::SparseMatrix(SparseMatrix & s):
size(s.size), 
numberOfRows(s.numberOfRows), 
numberOfColumns(s.numberOfColumns),
old_values_ref(s.new_values){
    //new_values = s.new_values;
}

/* SPM by reference */
SparseMatrix::SparseMatrix(int numberOfRows, std::vector<int> & values_ref_arg):
numberOfRows(numberOfRows),
old_values_ref(values_ref_arg){
    new_values.resize(values_ref_arg.size(), 1);
}


int SparseMatrix::GetNumberOfRows(){
    return numberOfRows;
}

