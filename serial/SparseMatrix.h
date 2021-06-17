#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H
#include <vector>
#include <sstream>      // std::stringstream
#include <iostream>

class SparseMatrix {
    public: 
        /* Will be useful for parallel builds */
        SparseMatrix(int size, int numberOfRows, int numberOfColumns);
        /* Will use AddEdge to build, thus size is dynamic */
        SparseMatrix(int numberOfRows, int numberOfColumns);
        /* Copy Constructor */
        SparseMatrix(const SparseMatrix & s);
        /* SpRef */
        SparseMatrix(const SparseMatrix & s, int edgesLeftToCover);
        virtual std::string toString() = 0;
        virtual void insertElements(const SparseMatrix & s) = 0;
        int numberOfRows, numberOfColumns, size;
        std::vector<int> values;
};

#endif