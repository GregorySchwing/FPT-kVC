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
        SparseMatrix(SparseMatrix & s);
        /* No optimization by reserving vectors, size and values must be set */
        SparseMatrix(int numberOfRows);
        /* SPM by reference */
        SparseMatrix(int numberOfRows, std::vector<int> & values_ref_arg);
        /* SpRef */
        SparseMatrix(const SparseMatrix & s, int edgesLeftToCover);

        int GetNumberOfRows();


        virtual std::string toString() = 0;
        int numberOfRows, numberOfColumns, size;
        // These are for the current matrix
        std::vector<int> & values_ref;
        // These are for the next matrix
        std::vector<int> values;
};

#endif