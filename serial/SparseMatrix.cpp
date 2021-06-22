#include "SparseMatrix.h"

/* Will be useful for parallel builds */
SparseMatrix::SparseMatrix(int size, int numberOfRows, int numberOfColumns):size(size), numberOfRows(numberOfRows), numberOfColumns(numberOfColumns){};
/* Will use AddEdge to build, thus size is dynamic */
SparseMatrix::SparseMatrix(int numberOfRows, int numberOfColumns):numberOfRows(numberOfRows), numberOfColumns(numberOfColumns){};
SparseMatrix::SparseMatrix(const SparseMatrix & s):size(s.size), numberOfRows(s.numberOfRows), numberOfColumns(s.numberOfColumns){
    values = s.values;
}
/* Induced Subgraph - SpRef - Only reserve so we only have to make 1 pass over the edges */
SparseMatrix::SparseMatrix(const SparseMatrix & s, int edgesLeftToCover):size(edgesLeftToCover), numberOfRows(s.numberOfRows), numberOfColumns(s.numberOfColumns){
    values.reserve(edgesLeftToCover);
    size = edgesLeftToCover;
}

/* No optimization by reserving vectors, size and values must be set */
SparseMatrix::SparseMatrix(){}

