

//#include "holder.cu"
#include "COO.cuh"
/*
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

*/


int main(int argc, char *argv[])
{
    int N = atoi(argv[1]);
    int numEntries = atoi(argv[2]);

    std::cout << "C1" << std::endl;
    COO c1(N, numEntries);
    std::cout << "C2" << std::endl;
    COO c2(N, numEntries);

    std::cout << "C1 inserted into C2" << std::endl;
    c2.insertElements(c1);
    std::cout << c2.toString() << std::endl;
}