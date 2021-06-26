#include "openmp/Graph.h"
#include "openmp/COO.h"
#include "openmp/CSR.h"
#include "openmp/DCSR.h"
#include "openmp/ParallelKernelization.h"
#include "openmp/ParallelBuss.h"
#include "openmp/ParallelB1.h"

//#include "gpu/COO.cuh"

int main(int argc, char *argv[])
{
    /* Num edges, N, N, random entries? */
    Graph g(10);
    int k = 4;
    ParallelKernelization sk(g, k);

}
