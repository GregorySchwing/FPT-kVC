#include "SparseMatrix.h"

/* Will be useful for parallel builds */
SparseMatrix::SparseMatrix(int size, int numberOfRows, int numberOfColumns):
size(size), 
numberOfRows(numberOfRows), 
numberOfColumns(numberOfColumns),
values_ref(values)
{};
/* Will use AddEdge to build, thus size is dynamic */
SparseMatrix::SparseMatrix(int numberOfRows, int numberOfColumns):
numberOfRows(numberOfRows), 
numberOfColumns(numberOfColumns),
values_ref(values)
{};
SparseMatrix::SparseMatrix(const SparseMatrix & s):
size(s.size), 
numberOfRows(s.numberOfRows), 
numberOfColumns(s.numberOfColumns),
values_ref(values){
    values = s.values;
}
/* Induced Subgraph - SpRef - Only reserve so we only have to make 1 pass over the edges */
SparseMatrix::SparseMatrix(const SparseMatrix & s, int edgesLeftToCover):
size(edgesLeftToCover), 
numberOfRows(s.numberOfRows), 
numberOfColumns(s.numberOfColumns),
values_ref(values){
    values.reserve(edgesLeftToCover);
    size = edgesLeftToCover;
}

/* No optimization by reserving vectors, size and values must be set */
SparseMatrix::SparseMatrix(int numberOfRows):
numberOfRows(numberOfRows),
values_ref(values){

}


/* SPM by reference */
SparseMatrix::SparseMatrix(int numberOfRows, std::vector<int> & values_ref_arg):
numberOfRows(numberOfRows),
values_ref(values_ref_arg){
    values.resize(values_ref_arg.size(), 1);
}

