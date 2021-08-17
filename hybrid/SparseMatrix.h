#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H
#include <vector>
#include <sstream>      // std::stringstream
#include <iostream>
#include <thrust/host_vector.h>

class SparseMatrix {
    public: 
        /* Will be useful for parallel builds */
        SparseMatrix(int size, int numberOfRows, int numberOfColumns);
        /* Will use AddEdge to build, thus size is dynamic */
        SparseMatrix(int numberOfRows, int numberOfColumns);

        SparseMatrix(SparseMatrix & s);
        /* Copy Constructor */
        SparseMatrix(const SparseMatrix & s);
        /* No optimization by reserving vectors, size and values must be set */
        SparseMatrix(int numberOfRows);

        SparseMatrix();


        /* SPM by reference */
        //SparseMatrix(int numberOfRows, thrust::host_vector<int> & values_ref_arg);

        int GetNumberOfRows();
        int GetSize();
        void SetNumberOfRows(int numberOfRows_arg);


        virtual std::string toString() = 0;
        int numberOfRows, numberOfColumns, size;
        // These are for the current matrix
        thrust::host_vector<int> * old_values_ref;
        // These are for the next matrix
        thrust::host_vector<int> new_values;
};

#endif