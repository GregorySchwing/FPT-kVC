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
typedef thrust::tuple<int,int,int> Int3;

struct cmp : public std::binary_function<Int3,Int3,bool>
{
    __host__ __device__
        bool operator()(const Int3& a, const Int3& b) const
        {
            if (thrust::get<0>(a) != thrust::get<0>(b))
                return thrust::get<0>(a) < thrust::get<0>(b);
            else 
                return thrust::get<1>(a) < thrust::get<1>(b);
        }
};

int main(int argc, char *argv[])
{
    int N = atoi(argv[1]);
    int numEntries = atoi(argv[2]);

/*
    int column_indices_a[numEntries], row_indices_a[numEntries], values_a[numEntries];
    int * column_indices_a_dev, * row_indices_a_dev, * values_a_dev;

    int column_indices_b[numEntries], row_indices_b[numEntries], values_b[numEntries];
    int * column_indices_b_dev, * row_indices_b_dev, * values_b_dev;

    cudaMalloc( (void**)&column_indices_a_dev, numEntries * sizeof(int) );
    cudaMalloc( (void**)&row_indices_a_dev, numEntries * sizeof(int) );
    cudaMalloc( (void**)&values_a_dev, numEntries * sizeof(int) );

    cudaMalloc( (void**)&column_indices_b_dev, numEntries * sizeof(int) );
    cudaMalloc( (void**)&row_indices_b_dev, numEntries * sizeof(int) );
    cudaMalloc( (void**)&values_b_dev, numEntries * sizeof(int) );

    int trialRow, trialCol;
    bool empty;
    for (int i = 0; i < numEntries; i++){
        do {
            empty = true;
            trialCol = std::rand() % N;
            trialRow = std::rand() %  N;
            for (int j = 0; j < i; j++){
                if (row_indices_a[j] == trialRow && column_indices_a[j] == trialCol)
                    empty = false;
            }
        } while (!empty);
        row_indices_a[i] = trialRow;
        column_indices_a[i] = trialCol;
        values_a[i] =  std::rand() % N + 1;
    }

    for (int i = 0; i < numEntries; i++){
        do {
            empty = true;
            trialCol = std::rand() % N;
            trialRow = std::rand() %  N;
            for (int j = 0; j < i; j++){
                if (row_indices_b[j] == trialRow && column_indices_b[j] == trialCol)
                    empty = false;
            }
        } while (!empty);
        row_indices_b[i] = trialRow;
        column_indices_b[i] = trialCol;
        values_b[i] =  std::rand() % N + 1;
    }

    cudaMemcpy(column_indices_a_dev, column_indices_a, numEntries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_indices_a_dev, row_indices_a, numEntries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(values_a_dev, values_a, numEntries * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(column_indices_b_dev, column_indices_b, numEntries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_indices_b_dev, row_indices_b, numEntries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(values_b_dev, values_b, numEntries * sizeof(int), cudaMemcpyHostToDevice);
*/
    // generate 16M random numbers on the host

    std::cout << "building vecs" << std::endl;
    thrust::device_vector<int> col_vec(numEntries);
    thrust::device_vector<int> row_vec(numEntries);
    thrust::device_vector<int> val_vec(numEntries);

    thrust::device_vector<int> dimensions(numEntries);
    thrust::device_vector<int> ones(numEntries);


    // fill dimensions with twos
    thrust::fill(dimensions.begin(), dimensions.end(), N);
    thrust::fill(ones.begin(), ones.end(), 1);

    std::cout << "generating vecs on gpu" << std::endl;

    thrust::generate(col_vec.begin(), col_vec.end(), rand);
    thrust::generate(row_vec.begin(), row_vec.end(), rand);
    thrust::generate(val_vec.begin(), val_vec.end(), rand);

    // compute Y = X mod 2
    std::cout << "transforming vecs on gpu" << std::endl;

    thrust::transform(col_vec.begin(), col_vec.end(), dimensions.begin(), col_vec.begin(), thrust::modulus<int>());
    thrust::transform(row_vec.begin(), row_vec.end(), dimensions.begin(), row_vec.begin(), thrust::modulus<int>());
    thrust::transform(val_vec.begin(), val_vec.end(), dimensions.begin(), val_vec.begin(), thrust::modulus<int>());
    thrust::transform(val_vec.begin(), val_vec.end(), ones.begin(), val_vec.begin(), thrust::plus<int>());

    // METHOD #1
    // Defining a zip_iterator type can be a little cumbersome ...
    std::cout << "creating tuples" << std::endl;

    typedef thrust::device_vector<int>::iterator                     IntIterator;
    typedef thrust::tuple<IntIterator, IntIterator, IntIterator> IntIteratorTuple;
    typedef thrust::zip_iterator<IntIteratorTuple>                   Int3Iterator;

    std::cout << "creating iterators" << std::endl;

    // Now we'll create some zip_iterators for A and B
    Int3Iterator A_first = thrust::make_zip_iterator(thrust::make_tuple(col_vec.begin(), row_vec.begin(), val_vec.begin()));
    Int3Iterator A_last  = thrust::make_zip_iterator(thrust::make_tuple(col_vec.end(),   row_vec.end(),   val_vec.end()));
    //Int3Iterator B_first = thrust::make_zip_iterator(thrust::make_tuple(B0.begin(), B1.begin(), B2.begin()));
        std::cout << "sorting" << std::endl;

    thrust::sort(A_first, A_last, cmp());

    std::cout << "creating host vecs" << std::endl;

    thrust::host_vector<int> col_vec_host(numEntries);
    thrust::host_vector<int> row_vec_host(numEntries);
    thrust::host_vector<int> val_vec_host(numEntries);

    col_vec_host = col_vec;
    row_vec_host = row_vec;
    val_vec_host = val_vec;

    std::stringstream ss;
    std::string myMatrix;
    ss << "\t\tCOO Matrix" << std::endl;
    for (int i = 0; i<numEntries; i++){
        ss << "\tcol " << i;
    }
    ss << std::endl;
    int row_index = 0;
    for (int i = 0; i < numEntries; i++){
        ss << "row " << i;
        for( int j = 0; j < numEntries; j++){
            if (row_vec_host[row_index] ==  i){
                if(j==col_vec_host[row_index]){
                    ss << "\t" << val_vec_host[row_index];
                    // Skip duplicate entries
                    while(row_vec_host[row_index] == i && j == col_vec_host[row_index]){
                        row_index++;
                    }
                } else {
                    ss << "\t" << 0;
                }
            } else {
                ss << "\t" << 0;
            }
        }        
        ss << std::endl;
    }
    ss << "Row indices" << std::endl;
    for(int i = 0; i< row_vec_host.size(); i++){
        ss << "\t" << row_vec_host[i];
    }
    ss << std::endl;
    ss << "Column indices" << std::endl;
    for(int i = 0; i< col_vec_host.size(); i++){
        ss << "\t" << col_vec_host[i];
    }
    ss << std::endl;
    ss << "values" << std::endl;
    for(int i = 0; i< val_vec_host.size(); i++){
        ss << "\t" << val_vec_host[i];
    }
    ss << std::endl;
    myMatrix = ss.str();

    std::cout << myMatrix << std::endl;
}