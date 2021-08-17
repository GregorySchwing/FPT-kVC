
#ifndef Parallel_B1_H
#define Parallel_B1_H

#include "Graph.h"
#include "ParallelKernelization.h"
#include <iostream>
#include <thrust/host_vector.h>


class ParallelB1 {
public:
void static PopulateTree(int treeSize, 
                                std::vector<Graph> & graphs,
                                thrust::host_vector<int> & answer);
void static PopulateTreeParallel(int treeSize, 
                                std::vector<Graph> & graphs,
                                thrust::host_vector<int> & answer);
int static PopulateTreeParallelLevelWise(int numberOfLevels, 
                                std::vector<Graph> & graphs,
                                thrust::host_vector<int> & answer);
void static PopulateTreeParallelAsymmetric(int treeSize, 
                                std::vector<Graph> & graphs,
                                thrust::host_vector<int> & answer);
int static GenerateChildren( Graph & child_g);
int static CalculateWorstCaseSpaceComplexity(int vertexCount);
long long static CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels);
void static TraverseUpTree(int index, 
                            std::vector<Graph> & graphs,
                            thrust::host_vector<int> & answer);


private:
    void static DFS(thrust::host_vector<int> & new_row_off,
                    thrust::host_vector<int> & new_col_ref,
                    thrust::host_vector<int> & new_vals_ref,
                    thrust::host_vector<int> & path, 
                    int rootVertex);
    int static GetRandomVertex(thrust::host_vector<int> & verticesRemaining);
    int static GetRandomOutgoingEdge(thrust::host_vector<int> & new_row_off,
                                    thrust::host_vector<int> & new_col_ref,
                                    thrust::host_vector<int> & new_values_ref,
                                    int v, 
                                    thrust::host_vector<int> & path);
    int static classifyPath(thrust::host_vector<int> & path);
    void static createVertexSetsForEachChild(std::vector< thrust::host_vector<int> > & childrensVertices, 
                                                int caseNumber, 
                                                thrust::host_vector<int> & path);
};

#endif
