#ifndef COO_CUH
#define COO_CUH
//#include "SparseMatrix.h"
#include <cstdlib> /* rand */
#include <vector>
#include <string>
#include <iostream>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>      // std::stringstream
// We'll use a 3-tuple to store our 3d vector type
// rows, cols, vals
//#include <cusp/array1d.h> 

class COO
{
public:
    COO(int dimensions, int numberOfEntries);
    std::string toString();
    void sortMyself();
    //bool getIsSorted() const { return isSorted; }
    void insertElements(COO & c);
    //COO& SpMV(COO & c);
    //int * column_indices, * row_indices, * values;
    int dimensions, numberOfEntries;
    thrust::host_vector<int> col_vec;
    thrust::host_vector<int> row_vec;
    thrust::host_vector<int> val_vec;

    thrust::host_vector<int> dimensions_vec;
    thrust::host_vector<int> ones;

    thrust::device_vector<int> col_vec_dev;
    thrust::device_vector<int> row_vec_dev;
    thrust::device_vector<int> val_vec_dev;
private:
//void wrapperFunction(int * column_indices, int * row_indices, int * values, int size);
//bool isSorted;
};
#endif