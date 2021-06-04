#include "SparseMatrix.h"

/* Will be useful for parallel builds */
SparseMatrix::SparseMatrix(int size, int numberOfRows, int numberOfColumns):size(size), numberOfRows(numberOfRows), numberOfColumns(numberOfColumns){};
/* Will use AddEdge to build, thus size is dynamic */
SparseMatrix::SparseMatrix(int numberOfRows, int numberOfColumns):numberOfRows(numberOfRows), numberOfColumns(numberOfColumns){};
SparseMatrix::SparseMatrix(const SparseMatrix & s):size(s.size), numberOfRows(s.numberOfRows), numberOfColumns(s.numberOfColumns){};