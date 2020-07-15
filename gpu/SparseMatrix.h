#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H
#include <vector>
#include <sstream>      // std::stringstream
#include <iostream>

class SparseMatrix {
    public: 
        SparseMatrix(int size, int numberOfRows, int numberOfColumns);
        SparseMatrix(const SparseMatrix & s);
        virtual std::string toString() = 0;
        virtual void insertElements(const SparseMatrix & s) = 0;
        int numberOfRows, numberOfColumns, size;
        int * values;
};

#endif