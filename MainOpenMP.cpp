#include "openmp/Graph.h"
#include "openmp/COO.h"
#include "openmp/CSR.h"
#include "openmp/ParallelKernelization.h"
#include "openmp/ParallelB1.h"
#include "common/CSVRange.h"
//#include "gpu/COO.cuh"

int main(int argc, char *argv[])
{
    /* Num edges, N, N, random entries? */
/*    std::cout << "Building G" << std::endl;
    Graph g(10);
    int k = 4;
    std::cout << "Building PK" << std::endl;
    ParallelKernelization sk(g, k);
    Graph * gPrime = new Graph(&g, sk.GetS());
    ParallelB1 pb1(gPrime, 
                k, 
                sk.GetS());
                //, gPrime->GetRemainingVerticesRef());
    pb1.IterateTreeStructure(&pb1);
*/
    char sep = ' ';
    std::ifstream       file("0.edges");
    for(auto& row: CSVRange(file, sep))
    {
        std::cout << "4th Element(" << row[0] << ")\n";
    }

}
