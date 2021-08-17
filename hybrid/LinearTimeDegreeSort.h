
#ifndef LINEAR_TIME_DEGREE_SORT_H
#define LINEAR_TIME_DEGREE_SORT_H

#include <vector>
#include <iostream>
#include <numeric>
#include <thrust/host_vector.h>

class LinearTimeDegreeSort {
public:
    LinearTimeDegreeSort(int numberOfRows, thrust::host_vector<int> & old_degree_ref);
    thrust::host_vector<int> & GetDegreeRef();
    thrust::host_vector<int> & GetVertexKeyRef();

private:
    thrust::host_vector<int> B_row_indices;
    thrust::host_vector<int> B_column_indices; 
    thrust::host_vector<int> B_values; 
    thrust::host_vector<int> C;

    void CountingSortSerial(int max,
                    const thrust::host_vector<int> & A_row_indices,
                    thrust::host_vector<int> & A_column_indices,
                    thrust::host_vector<int> & A_values,
                    thrust::host_vector<int> & B_row_indices_ref,
                    thrust::host_vector<int> & B_column_indices_ref,
                    thrust::host_vector<int> & B_values_ref,
                    thrust::host_vector<int> & C_ref);
};
#endif