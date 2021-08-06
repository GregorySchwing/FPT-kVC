#include "openmp/Graph.h"
#include "openmp/COO.h"
#include "openmp/CSR.h"
#include "openmp/ParallelKernelization.h"
#include "openmp/ParallelB1.h"
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
    std::cout << "Building G" << std::endl;
    //Graph g("0.edges");
    std::shared_ptr<Graph> g = std::make_shared<Graph>("0.edges");
    int k = 4;
    std::cout << "Building PK" << std::endl;
    ParallelKernelization sk(g, k);
    for (int i = k; i < g->GetVertexCount(); ++i){
    // If (noSolutionExists)
        if (sk.TestAValueOfK(i))
            continue;
        else {
            sk.PrintS();            

            std::shared_ptr<Graph> gPrime = std::make_shared<Graph>(g, sk.GetS()); 
            std::shared_ptr<ParallelB1> pb1 = std::make_shared<ParallelB1>(gPrime, 
                                                                        k, 
                                                                        sk.GetS());
        }
    }
}
