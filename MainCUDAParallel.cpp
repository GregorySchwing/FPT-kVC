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

#include <iostream>
#include <random>
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
    int numberOfRows = g.GetVertexCount();
    int edgesLeftToCover = g.GetEdgesLeftToCover();
    int * host_levels = new int[numberOfRows];

    int * new_colors = new int[numberOfRows];
    int * new_U = new int[numberOfRows];
    int * new_Pred = new int[numberOfRows];
    int * new_color_finished = new int[numberOfRows];

    int * global_row_offsets_dev_ptr; // size N + 1
    int * global_columns_dev_ptr; // size M
    int * global_values_dev_ptr; // on or off, size M
    int * global_degrees_dev_ptr; // size N, used for inducing the subgraph
    int * global_triangle_remaining_boolean; // size |W|, where W is the subset of V contained in a triangle, used for finding MIS of triangles
    int * triangle_row_offsets_array_dev;

    // Vertex, Cols, Edge(on/off)
    cudaMalloc( (void**)&global_row_offsets_dev_ptr, (numberOfRows+1) * sizeof(int) );
    cudaMalloc( (void**)&triangle_row_offsets_array_dev, (numberOfRows+1) * sizeof(int) );
    cudaMalloc( (void**)&global_columns_dev_ptr, edgesLeftToCover * sizeof(int) );
    cudaMalloc( (void**)&global_values_dev_ptr, edgesLeftToCover * sizeof(int) );
    cudaMalloc( (void**)&global_degrees_dev_ptr, (numberOfRows+1) * sizeof(int) );

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    int * new_row_offsets = new int[numberOfRows+1];
    int * new_cols = new int[edgesLeftToCover];
    int * new_vals = new int[edgesLeftToCover];

    CallInduceSubgraph(g, 
                    global_row_offsets_dev_ptr,
                    global_columns_dev_ptr,
                    global_values_dev_ptr,
                    global_degrees_dev_ptr,
                    new_row_offsets,
                    new_cols,
                    new_vals);

    int * triangle_counter_host  = new int[numberOfRows+1];
    int * triangle_counter_dev;
    cudaMalloc( (void**)&triangle_counter_dev, (numberOfRows+1) * sizeof(int) );
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
    int * triangle_row_offsets_array_host = new int[numberOfRows+1];
    int * triangle_candidates_a_dev;
    int * triangle_candidates_b_dev;
    int * triangle_candidates_a_host;
    int * triangle_candidates_b_host;
    int numberOfTriangles_host;
    CallCountTriangles(
                        numberOfRows,
                        edgesLeftToCover,
                        &numberOfTriangles_host,
                        global_row_offsets_dev_ptr,
                        global_columns_dev_ptr,
                        new_row_offsets,
                        new_cols,
                        triangle_counter_host,
                        triangle_counter_dev,
                        triangle_row_offsets_array_host,
                        triangle_row_offsets_array_dev);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
    printf("number of triangles %d\n", numberOfTriangles_host);
    
    std::cout << "Triangle row offs" << std::endl;
    for (int i = 0; i <= numberOfRows; ++i){
        printf("vertex %d degree %d\n",i,triangle_row_offsets_array_host[i]);
    }
    std::cout << std::endl;
    cudaMalloc( (void**)&triangle_candidates_a_dev, numberOfTriangles_host * sizeof(int) );
    cudaMalloc( (void**)&triangle_candidates_b_dev, numberOfTriangles_host * sizeof(int) );
    triangle_candidates_a_host = new int[numberOfTriangles_host];
    triangle_candidates_b_host = new int[numberOfTriangles_host];

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    CallSaveTriangles(numberOfRows,
                        numberOfTriangles_host,
                        global_row_offsets_dev_ptr,
                        global_columns_dev_ptr,
                        triangle_row_offsets_array_host,
                        triangle_row_offsets_array_dev,
                        triangle_candidates_a_host,
                        triangle_candidates_b_host,
                        triangle_candidates_a_dev,
                        triangle_candidates_b_dev);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);    

    std::cout << "Number of Triangles" << std::endl;
    for (int i = 0; i < numberOfRows; ++i){
        std::cout << triangle_counter_host[i] << " ";
    }
    std::cout << std::endl;

    CallDisjointSetTriangles(
                                numberOfRows,
                                global_row_offsets_dev_ptr,
                                global_columns_dev_ptr,
                                triangle_row_offsets_array_dev,
                                triangle_counter_host,
                                triangle_counter_dev,
                                triangle_candidates_a_dev,
                                triangle_candidates_b_dev);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    cudaMemcpy(&triangle_counter_host[0], &triangle_counter_dev[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    std::cout << "Number of Triangles After conflict resolution" << std::endl;
    int sum = 0;
    for (int i = 0; i < numberOfRows; ++i){
        std::cout << triangle_counter_host[i] << " ";
        sum += triangle_counter_host[i];
    }
    std::cout << std::endl;
   std::cout << "Percentage of graph partitioned into a triangle" << std::endl;
    double percentTri = ((double)sum)/((double)numberOfRows) * 100.00;
    printf("%.2f %%\n", percentTri);



/*
    cudaMalloc( (void**)&global_triangle_remaining_boolean, numberOfTriangles_host * sizeof(int) );

    CallMIS(g,
            global_row_offsets_dev_ptr,
            global_columns_dev_ptr,
            global_values_dev_ptr,
            global_degrees_dev_ptr,
            global_triangle_remaining_boolean);
*/
    // Step 1
    //SSSPAndBuildDepthCSR(g, root, host_levels, new_row_offsets, new_cols, new_colors, new_U, new_Pred, new_color_finished);
    // Step 2
    //EnumerateSearchTree(g, new_row_offsets, new_cols, new_colors, new_color_finished);

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 655); // define the range

    int * new_colors_randomized = new int[numberOfRows];
    int * new_colors_mapper = new int[numberOfRows];

    for(int n=0; n<numberOfRows; ++n){
        new_colors_mapper[n] = distr(gen); // generate numbers
    }

    for(int n=0; n<numberOfRows; ++n){
        new_colors_randomized[n] = new_colors_mapper[new_colors[n]]; // generate numbers
    }

    std::string name = "main";
    std::string filenameGraph = "BFS";
    bool isDirected = false;
    DotWriter::RootGraph gVizWriter(isDirected, name);
    std::string subgraph1 = "BFS";
    std::string subgraph2 = "graph";
    std::string subgraph3 = "pred";

    DotWriter::Subgraph * bfs = gVizWriter.AddSubgraph(subgraph1);
    DotWriter::Subgraph * graph = gVizWriter.AddSubgraph(subgraph2);
    DotWriter::Subgraph * pred = gVizWriter.AddSubgraph(subgraph3);

    std::map<std::string, DotWriter::Node *> bfsMap;    

    std::map<std::string, DotWriter::Node *> nodeMap;    


    // Since the graph doesnt grow uniformly, it is too difficult to only copy the new parts..
    for (int i = 0; i < numberOfRows; ++i){
        std::string node1Name = std::to_string(i);
        std::map<std::string, DotWriter::Node *>::const_iterator nodeIt1 = nodeMap.find(node1Name);
        if(nodeIt1 == nodeMap.end()) {
            nodeMap[node1Name] = graph->AddNode(node1Name);
            if(new_color_finished[new_colors[i]]){
                nodeMap[node1Name]->GetAttributes().SetColor(DotWriter::Color::e(new_colors_randomized[i]));
                nodeMap[node1Name]->GetAttributes().SetFillColor(DotWriter::Color::e(new_colors_randomized[i]));
                nodeMap[node1Name]->GetAttributes().SetStyle("filled");
            }
        }
        for (int j = new_row_offsets[i]; j < new_row_offsets[i+1]; ++j){
            if (i < new_cols[j]){
                std::string node2Name = std::to_string(new_cols[j]);
                std::map<std::string, DotWriter::Node *>::const_iterator nodeIt2 = nodeMap.find(node2Name);
                if(nodeIt2 == nodeMap.end()) {
                    nodeMap[node2Name] = graph->AddNode(node2Name);
                    if(new_color_finished[new_colors[new_cols[j]]]){
                        nodeMap[node2Name]->GetAttributes().SetColor(DotWriter::Color::e(new_colors_randomized[new_cols[j]]));
                        nodeMap[node2Name]->GetAttributes().SetFillColor(DotWriter::Color::e(new_colors_randomized[new_cols[j]]));
                        nodeMap[node2Name]->GetAttributes().SetStyle("filled");
                    }
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
            bfsMap[node2Name]->GetAttributes().SetColor(DotWriter::Color::e(new_colors_randomized[i]));
            bfsMap[node2Name]->GetAttributes().SetFillColor(DotWriter::Color::e(new_colors_randomized[i]));
            bfsMap[node2Name]->GetAttributes().SetStyle("filled");
        }  
        bfs->AddEdge(bfsMap[node1Name], bfsMap[node2Name], std::to_string(new_U[i])); 
    }



    gVizWriter.WriteToFile(filenameGraph);
    std::cout << "finished" << std::endl;
}
