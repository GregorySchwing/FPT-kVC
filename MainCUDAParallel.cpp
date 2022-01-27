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
// For viz
#include "../lib/DotWriter/lib/DotWriter.h"
#include "../lib/DotWriter/lib/Enums.h"

#include <map>
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
    std::string filename = "/home6/greg/FPT-kVC/25_nodes.csv";
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
    int numberOfRows = g.GetRemainingVertices().size(); 
    int * host_levels = new int[numberOfRows];
    int * new_row_offsets = new int[numberOfRows+1];
    int * new_cols = new int[g.GetEdgesLeftToCover()];
    int * new_colors = new int[numberOfRows];
    int * new_U = new int[numberOfRows];

    CallPopulateTree(g, root, host_levels, new_row_offsets, new_cols, new_colors, new_U);

    std::string name = "main";
    std::string filenameGraph = "BFS";
    bool isDirected = false;
    DotWriter::RootGraph gVizWriter(isDirected, name);
    std::string subgraph1 = "BFS";
    std::string subgraph2 = "graph";

    DotWriter::Subgraph * bfs = gVizWriter.AddSubgraph(subgraph1);
    DotWriter::Subgraph * graph = gVizWriter.AddSubgraph(subgraph2);

    std::map<std::string, DotWriter::Node *> bfsMap;    

    std::map<std::string, DotWriter::Node *> nodeMap;    

    // Since the graph doesnt grow uniformly, it is too difficult to only copy the new parts..
    for (int i = 0; i < numberOfRows; ++i){
        std::string node1Name = std::to_string(i);
        std::map<std::string, DotWriter::Node *>::const_iterator nodeIt1 = nodeMap.find(node1Name);
        if(nodeIt1 == nodeMap.end()) {
            nodeMap[node1Name] = graph->AddNode(node1Name);
            //nodeMap[node1Name]->GetAttributes().SetColor(DotWriter::Color::e(1+new_colors[i]));
            //nodeMap[node1Name]->GetAttributes().SetFillColor(DotWriter::Color::e(1+new_colors[i]));
            //nodeMap[node1Name]->GetAttributes().SetStyle("filled");
        }
        for (int j = new_row_offsets[i]; j < new_row_offsets[i+1]; ++j){
            if (i < new_cols[j]){
                std::string node2Name = std::to_string(new_cols[j]);
                std::map<std::string, DotWriter::Node *>::const_iterator nodeIt2 = nodeMap.find(node2Name);
                if(nodeIt2 == nodeMap.end()) {
                    nodeMap[node2Name] = graph->AddNode(node2Name);
                    //nodeMap[node2Name]->GetAttributes().SetColor(DotWriter::Color::e(1+new_colors[new_cols[j]]));
                    //nodeMap[node2Name]->GetAttributes().SetFillColor(DotWriter::Color::e(1+new_colors[new_cols[j]]));
                    //nodeMap[node2Name]->GetAttributes().SetStyle("filled");
                }  
                //graph->AddEdge(nodeMap[node1Name], nodeMap[node2Name], std::to_string(host_levels[i]));
                graph->AddEdge(nodeMap[node1Name], nodeMap[node2Name]); 
 
            }
        }
    }

    std::string node1Name = std::to_string(root);
    std::map<std::string, DotWriter::Node *>::const_iterator nodeIt1 = bfsMap.find(node1Name);
    if(nodeIt1 == bfsMap.end()) {
        bfsMap[node1Name] = bfs->AddNode(node1Name);
    }
    for (int i = 0; i < numberOfRows; ++i){
        std::string node2Name = std::to_string(i);
        std::map<std::string, DotWriter::Node *>::const_iterator nodeIt2 = bfsMap.find(node2Name);
        if(nodeIt2 == bfsMap.end()) {
            bfsMap[node2Name] = bfs->AddNode(node2Name);
            //bfsMap[node2Name]->GetAttributes().SetColor(DotWriter::Color::e(1+new_colors[i]));
            //bfsMap[node2Name]->GetAttributes().SetFillColor(DotWriter::Color::e(1+new_colors[i]));
            //bfsMap[node2Name]->GetAttributes().SetStyle("filled");
        }  
        bfs->AddEdge(bfsMap[node1Name], bfsMap[node2Name], std::to_string(new_U[i])); 
    }

    gVizWriter.WriteToFile(filenameGraph);
    std::cout << "finished" << std::endl;
}
