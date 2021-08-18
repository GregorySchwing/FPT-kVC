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
//    std::string filename = "small.csv";
    std::string filename = "25_nodes.csv";
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

    CallPopulateTree(endingLevel - startingLevel, 
                    graphs[0]);
    //thrust::device_vector< int > verticesToIncludeInCover_dev(g.GetVertexCount()*treeSize);
    //thrust::device_vector< int > verticesRemaining_dev(g.GetVertexCount()*treeSize);
    //thrust::device_vector< int > hasntBeenRemoved_dev(g.GetVertexCount()*treeSize);

    /*
    std::vector< Graph > graphs(treeSize, Graph(g));
    thrust::host_vector<int> mpt;
    graphs[0].InitGPrime(g, mpt);
    graphs[0].SetVerticesToIncludeInCover(g.GetVerticesThisGraphIncludedInTheCover());
    //std::swap(graphs[0], gPrime);
    thrust::host_vector<int> answer;
    //ParallelB1::PopulateTree(treeSize, graphs, answer);
    int result = ParallelB1::PopulateTreeParallelLevelWise(numberOfLevels, graphs, answer);
    std::cout << std::endl;
    if (result != -1){
        ParallelB1::TraverseUpTree(result, graphs, answer);    
        std::cout << std::endl;
        std::cout << "Found an answer" << std::endl;
        for (auto & v: answer)
            std::cout << v << " ";
        std::cout << std::endl;
    }

    COO coordinateFormatTest;
    coordinateFormatTest.BuildTheExampleCOO();
//    coordinateFormatTest.BuildCOOFromFile(filename);
    CSR csrTest(coordinateFormatTest);
    Graph gTest(csrTest);
    gTest.InitG(gTest, answer);
    std::cout << "Edges remaining in original graph after removing answer : " << gTest.GetEdgesLeftToCover() << std::endl;
    gTest.PrintEdgesRemaining();    
    */
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
    thrust::host_vector<int> answer;
    int kCovSize = binarySearch(answer,
                                minK,
                                maxK,
                                sk,
                                &g);
    */
}
