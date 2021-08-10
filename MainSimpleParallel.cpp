#include "simpleParallel/Graph.h"
#include "simpleParallel/ParallelKernelization.h"
#include <vector>
#include <memory>
#include "simpleParallel/ConnectednessTest.h"
#include "simpleParallel/ParallelB1.h"

//#include "simpleParallel/COO.h"
//#include "simpleParallel/CSR.h"
//#include "gpu/COO.cuh"


int main(int argc, char *argv[])
{
    /* Num edges, N, N, random entries? */
    std::cout << "Building G" << std::endl;
    //Graph g("small.csv");
    COO coordinateFormat;
    std::string filename = "small.csv";
    /* Eventually replace this with an initialization from file */
    coordinateFormat.BuildCOOFromFile(filename);
    //coordinateFormat.BuildTheExampleCOO();

    coordinateFormat.SetVertexCountFromEdges();
    std::vector< std::vector<int> > vectorOfConnectedComponents;
    ConnectednessTest ct(coordinateFormat, vectorOfConnectedComponents);
    if (vectorOfConnectedComponents.size() > 1){
        std::cout << "Graph isn't connected" << std::endl;
    } else {
        std::cout << "Graph is connected" << std::endl;
    }
    CSR csr(coordinateFormat);
    Graph g(csr);
    std::cout << g.GetRemainingVerticesRef()[0] << std::endl;
    int k = 19;
    ParallelKernelization sk(g, k);
    sk.TestAValueOfK(k);
    bool noSolutionExists = sk.EdgeCountKernel();
    if(noSolutionExists){
        std::cout << "|G'(E)| > k*k', no solution exists" << std::endl;
    } else{
        std::cout << "|G'(E)| <= k*k', a solution may exist" << std::endl;
    }
    Graph & gPrime = sk.GetGPrime(); 
    std::vector< Graph > graphs(5, Graph(gPrime));
    ParallelB1::GenerateChildren(gPrime);
    // ParallB1 Call to get gPrime's children
    // Store in test
    //graphs[0].InitGNPrime(gPrime, test);
/* 

    //Graph g(10);
    //bool exists = pb1.IterateTreeStructure(&pb1, answer);
    //if (exists){
    //    for (auto & v : answer)
    //        std::cout << v << " ";
    //    std::cout << std::endl;
    //} 
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