

//#include "holder.cu"
#include "COO.cuh"
#include "CSR.cuh"


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

    std::cout << "Creating CSR" << std::endl;
    CSR c3(c2);

}