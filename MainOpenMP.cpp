#include "openmp/Graph.h"
#include "openmp/COO.h"
#include "openmp/CSR.h"
#include "openmp/ParallelKernelization.h"
#include "openmp/ParallelB1.h"

//#include "gpu/COO.cuh"

int main(int argc, char *argv[])
{
    /* Num edges, N, N, random entries? */
    std::cout << "Building G" << std::endl;
    Graph g(10);
    int k = 4;
    std::cout << "Building PK" << std::endl;
    ParallelKernelization sk(g, k);
    Graph * gPrime = new Graph(&g, sk.GetS());

    ParallelB1(gPrime, 
                sk.GetKPrime(), 
                sk.GetS(), 
                gPrime->GetRemainingVerticesRef());
    //ParallelB1(sk.GetKPrime(), &sk);
}
