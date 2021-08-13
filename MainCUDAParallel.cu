#include <cuda.h>
#include "hybrid/Graph.h"
#include "hybrid/ParallelKernelization.h"
#include <vector>
#include <memory>
#include "hybrid/ConnectednessTest.h"
#include "hybrid/ParallelB1.h"
#include <unistd.h>

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
//    std::string filename = "25_nodes.csv";
//    coordinateFormat.BuildCOOFromFile(filename);
//    coordinateFormat.BuildTheExampleCOO();
    coordinateFormat.BuildCycleCOO();

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
    int k = 4;
    ParallelKernelization sk(g, k);
    sk.TestAValueOfK(k);
    bool noSolutionExists = sk.EdgeCountKernel();
    if(noSolutionExists){
        std::cout << "|G'(E)| > k*k', no solution exists" << std::endl;
    } else{
        std::cout << "|G'(E)| <= k*k', a solution may exist" << std::endl;
    }
    //int treeSize = 200000;
    int numberOfLevels = 5;
    long long treeSize = ParallelB1::CalculateSpaceForDesiredNumberOfLevels(numberOfLevels);
    long long expandedData = g.GetEdgesLeftToCover();
    long long condensedData = g.GetVertexCount();
    long long sizeOfSingleGraph = expandedData*2*sizeof(int) + 2*condensedData*sizeof(int);
    long long totalMem = sizeOfSingleGraph * treeSize;
    long long memAvail = getTotalSystemMemory();
    std::cout << "You are about to allocate " << double(totalMem)/1024/1024/1024 << " GB" << std::endl;
    std::cout << "Your system has " << double(memAvail)/1024/1024/1024 << " GB available" << std::endl;
    do 
    {
        std::cout << '\n' << "Press a key to continue...; ctrl-c to terminate";
    } while (std::cin.get() != '\n');
    std::vector< Graph > graphs(treeSize, Graph(g));
    std::vector<int> mpt;
    graphs[0].InitGPrime(g, mpt);
    graphs[0].SetVerticesToIncludeInCover(g.GetVerticesThisGraphIncludedInTheCover());
    //std::swap(graphs[0], gPrime);
    std::vector<int> answer;
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
