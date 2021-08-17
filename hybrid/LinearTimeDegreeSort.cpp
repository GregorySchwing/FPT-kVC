#include "LinearTimeDegreeSort.h"

LinearTimeDegreeSort::LinearTimeDegreeSort(int numberOfRows, thrust::host_vector<int> & old_degree_ref){
    // Sorted as pairs (deg_0, v_0) , (deg_1, v_1) ...
    // Could use thrust
    thrust::host_vector<int> vertexKeys(numberOfRows);
    std::iota (std::begin(vertexKeys), std::end(vertexKeys), 0); // Fill with 0, 1, ..., 99.

    int max_degree = 0;
    for (int i = 0; i < numberOfRows; ++i){
        if (old_degree_ref[i] > max_degree)
            max_degree = old_degree_ref[i];
    }

    std::cout << "Max : " << max_degree << std::endl;

    B_row_indices.resize(numberOfRows);
    B_column_indices.resize(numberOfRows);
    B_values.resize(numberOfRows);

    C.resize(max_degree+1, 0);

    // Sort by degree
    CountingSortSerial( max_degree,
                        old_degree_ref,
                        vertexKeys,
                        vertexKeys,
                        B_row_indices,
                        B_column_indices,
                        B_values,
                        C);

}

thrust::host_vector<int> & LinearTimeDegreeSort::GetDegreeRef(){
    return B_row_indices;
}

thrust::host_vector<int> & LinearTimeDegreeSort::GetVertexKeyRef(){
    return B_column_indices;
}


void LinearTimeDegreeSort::CountingSortSerial(int max,
                        const thrust::host_vector<int> & A_row_indices,
                        thrust::host_vector<int> & A_column_indices,
                        thrust::host_vector<int> & A_values,
                        thrust::host_vector<int> & B_row_indices_ref,
                        thrust::host_vector<int> & B_column_indices_ref,
                        thrust::host_vector<int> & B_values_ref,
                        thrust::host_vector<int> & C_ref)
{
    for (int i = 0; i < A_row_indices.size(); ++i){
        ++C_ref[A_row_indices[i]];
    }

    std::cout << "C[i] now contains the number elements equal to i." << std::endl;
    for (int i = 1; i < max+1; ++i){
        C_ref[i] = C_ref[i] + C_ref[i-1];
    }

    std::cout << "C[i] now contains the number of elements less than or equal to i." << std::endl;
    /* C_ref[A_row_indices[i]]]-1 , because the values of C_ref are from [1, n] -> [0,n)*/
    for (int i = B_row_indices.size()-1; i >= 0; --i){
        B_row_indices_ref[C_ref[A_row_indices[i]]-1] = A_row_indices[i];
        B_column_indices_ref[C_ref[A_row_indices[i]]-1] = A_column_indices[i];
        B_values_ref[C_ref[A_row_indices[i]]-1] = A_values[i];
        --C_ref[A_row_indices[i]];
    }
}