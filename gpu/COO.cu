#include "COO.cuh"
#include <math.h>       /* floor */
#include <cuda.h>

//static void coo(int N, int numberOfEntries);


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


COO::COO(int dimensions, int numberOfEntries): dimensions(dimensions), numberOfEntries(numberOfEntries){
    
// allocate storage for 4 values (uninitialized)    
    cusp::array1d<float, cusp::host_memory> A(4, -1.0f); 

    std::cout << "building vecs" << std::endl;
    col_vec.resize(numberOfEntries);
    row_vec.resize(numberOfEntries);
    val_vec.resize(numberOfEntries);

    dimensions_vec.resize(numberOfEntries);
    ones.resize(numberOfEntries);


    // fill dimensions vector with Ns
    thrust::fill(dimensions_vec.begin(), dimensions_vec.end(), dimensions);
    thrust::fill(ones.begin(), ones.end(), 1);

    std::cout << "generating vecs on host" << std::endl;

    thrust::generate(col_vec.begin(), col_vec.end(), rand);
    thrust::generate(row_vec.begin(), row_vec.end(), rand);
    thrust::generate(val_vec.begin(), val_vec.end(), rand);

    // compute Y = X mod N
    std::cout << "transforming vecs on host" << std::endl;

    thrust::transform(col_vec.begin(), col_vec.end(), dimensions_vec.begin(), col_vec.begin(), thrust::modulus<int>());
    thrust::transform(row_vec.begin(), row_vec.end(), dimensions_vec.begin(), row_vec.begin(), thrust::modulus<int>());
    thrust::transform(val_vec.begin(), val_vec.end(), dimensions_vec.begin(), val_vec.begin(), thrust::modulus<int>());
    thrust::transform(val_vec.begin(), val_vec.end(), ones.begin(), val_vec.begin(), thrust::plus<int>());

    sortMyself();

    std::cout << toString() << std::endl;
}

void COO::insertElements(COO & c){
    row_vec.insert(row_vec.end(), c.row_vec.begin(), c.row_vec.end());
    col_vec.insert(col_vec.end(), c.col_vec.begin(), c.col_vec.end());
    val_vec.insert(val_vec.end(), c.val_vec.begin(), c.val_vec.end());
    numberOfEntries+=c.numberOfEntries;
    sortMyself();
}


void COO::sortMyself(){
    std::cout << "copying vecs from host to device" << std::endl;

    col_vec_dev = col_vec;
    row_vec_dev = row_vec;
    val_vec_dev = val_vec;

    // METHOD #1
    // Defining a zip_iterator type can be a little cumbersome ...
    std::cout << "creating tuples" << std::endl;

    typedef thrust::device_vector<int>::iterator                     IntIterator;
    typedef thrust::tuple<IntIterator, IntIterator, IntIterator> IntIteratorTuple;
    typedef thrust::zip_iterator<IntIteratorTuple>                   Int3Iterator;

    std::cout << "creating iterators" << std::endl;

    // Now we'll create some zip_iterators for A and B
    Int3Iterator A_first = thrust::make_zip_iterator(thrust::make_tuple(row_vec_dev.begin(), col_vec_dev.begin(), val_vec_dev.begin()));
    Int3Iterator A_last  = thrust::make_zip_iterator(thrust::make_tuple(row_vec_dev.end(),   col_vec_dev.end(),   val_vec_dev.end()));
    //Int3Iterator B_first = thrust::make_zip_iterator(thrust::make_tuple(B0.begin(), B1.begin(), B2.begin()));
    std::cout << "sorting" << std::endl;

    thrust::sort(A_first, A_last, cmp());

    std::cout << "copying back to host vecs" << std::endl;

    col_vec = col_vec_dev;
    row_vec = row_vec_dev;
    val_vec = val_vec_dev;
}

std::string COO::toString(){
    std::stringstream ss;
    std::string myMatrix;
    ss << "\t\tCOO Matrix" << std::endl;
    for (int i = 0; i<dimensions; i++){
        ss << "\tcol " << i;
    }
    ss << std::endl;
    int row_index = 0;
    for (int i = 0; i < dimensions; i++){
        ss << "row " << i;
        for( int j = 0; j < dimensions; j++){
            if (row_vec[row_index] ==  i){
                if(j==col_vec[row_index]){
                    ss << "\t" << val_vec[row_index];
                    // Skip duplicate entries
                    while(row_vec[row_index] == i && j == col_vec[row_index]){
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
    for(int i = 0; i< row_vec.size(); i++){
        ss << "\t" << row_vec[i];
    }
    ss << std::endl;
    ss << "Column indices" << std::endl;
    for(int i = 0; i< col_vec.size(); i++){
        ss << "\t" << col_vec[i];
    }
    ss << std::endl;
    ss << "values" << std::endl;
    for(int i = 0; i< val_vec.size(); i++){
        ss << "\t" << val_vec[i];
    }
    ss << std::endl;
    myMatrix = ss.str();

    return myMatrix;
}