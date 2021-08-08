
#ifndef LINEAR_TIME_DEGREE_SORT_H
#define LINEAR_TIME_DEGREE_SORT_H

#include <vector>
#include <iostream>
#include <numeric>

class LinearTimeDegreeSort {
public:
    LinearTimeDegreeSort(int numberOfRows, const std::vector<int> & old_degree_ref);
    std::vector<int> & GetDegreeRef();
    std::vector<int> & GetVertexKeyRef();

private:
    std::vector<int> B_row_indices;
    std::vector<int> B_column_indices; 
    std::vector<int> B_values; 
    std::vector<int> C;

    void CountingSortSerial(int max,
                    std::vector<int> & A_row_indices,
                    std::vector<int> & A_column_indices,
                    std::vector<int> & A_values,
                    std::vector<int> & B_row_indices_ref,
                    std::vector<int> & B_column_indices_ref,
                    std::vector<int> & B_values_ref,
                    std::vector<int> & C_ref);
};
#endif