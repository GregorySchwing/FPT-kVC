#include <cuda.h>
#include "hybrid/Graph.h"
#include "hybrid/ParallelKernelization.h"
#include <vector>
#include <memory>
#include "hybrid/ConnectednessTest.h"
#include "hybrid/ParallelB1.h"
#include <unistd.h>
#ifdef FPT_CUDA
#include "ParallelB1_GPU.cuh"
#endif
unsigned long long getTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}

int main(int argc, char *argv[])
{
    /* Num edges, N, N, random entries? */
    std::cout << "Building G" << std::endl;
    //Graph g("small.csv");
    COO coordinateFormat;
    //std::string filename = "small.csv";
    //std::string filename = "simulated_blockmodel_graph_50_nodes.csv";
    std::string filename = "25_nodes.csv";
    //std::string filename = "pendants.csv";

    coordinateFormat.BuildCOOFromFile(filename);
//    coordinateFormat.BuildTheExampleCOO();
    //coordinateFormat.BuildCycleCOO();

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
    //int k = 15;
    int k = 9;
    ParallelKernelization sk(g, k);
    sk.TestAValueOfK(k);
    bool noSolutionExists = sk.EdgeCountKernel();
    if(noSolutionExists){
        std::cout << "|G'(E)| > k*k', no solution exists" << std::endl;
    } else{
        std::cout << "|G'(E)| <= k*k', a solution may exist" << std::endl;
    }

    int startingLevel = 0;
    int endingLevel = 5;
    std::vector< Graph > graphs(1, Graph(g));
    thrust::host_vector<int> mpt;
    graphs[0].InitGPrime(g, mpt);
    thrust::host_vector<int> & newCols = graphs[0].GetCSR().GetNewColRef();
    std::cout << "cpu newCols" << std::endl;
    for (auto & v: newCols)
        std::cout << v << " ";
    std::cout << std::endl;
    g.RemoveDegreeZeroVertices();
    int root = 0;
    CallPopulateTree(g, root);
}
