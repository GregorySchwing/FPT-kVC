#include "SparseMatrix.h"

SparseMatrix::SparseMatrix(int size, int numberOfRows, int numberOfColumns):size(size), numberOfRows(numberOfRows), numberOfColumns(numberOfColumns){};
SparseMatrix::SparseMatrix(const SparseMatrix & s):size(s.size), numberOfRows(s.numberOfRows), numberOfColumns(s.numberOfColumns){};