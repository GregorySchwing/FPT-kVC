#include "simpleParallel/Graph.h"
#include "simpleParallel/ParallelKernelization.h"
#include <vector>
#include <memory>
//#include "simpleParallel/COO.h"
//#include "simpleParallel/CSR.h"
//#include "gpu/COO.cuh"


int main(int argc, char *argv[])
{
    /* Num edges, N, N, random entries? */
    std::cout << "Building G" << std::endl;
    //Graph g("small.csv");
    Graph * g = new Graph("small.csv");
    int k = 15;
    ///ParallelKernelization sk(g, k);
    std::vector< Graph* > graphs;
    graphs.resize(5);
    graphs[0] = g;
    graphs[1] = new Graph(graphs[0]);
    //Graph g(10);
    //bool exists = pb1.IterateTreeStructure(&pb1, answer);
    //if (exists){
    //    for (auto & v : answer)
    //        std::cout << v << " ";
    //    std::cout << std::endl;
    //}
/*  
    std::cout << "Building G" << std::endl;
    Graph g("0.edges");
    int k = 4;
    std::cout << "Building PK" << std::endl;
    ParallelKernelization sk(g, k);
    int minK = 0;
    int maxK = g.GetVertexCount();
    for (int i = k; i < g.GetVertexCount(); ++i){
        // If (noSolutionExists)
        // If (Also clears and sets S if a sol'n could exist)
        if (sk.TestAValueOfK(i))
            continue;
        else{
            minK = i;
            break;
        }
    }
    std::cout << "Found min K : " << minK << std::endl;
    std::vector<int> answer;
    int kCovSize = binarySearch(answer,
                                minK,
                                maxK,
                                sk,
                                &g);
    */
}