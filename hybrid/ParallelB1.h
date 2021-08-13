
#ifndef Parallel_B1_H
#define Parallel_B1_H

#include "Graph.h"
#include "ParallelKernelization.h"
#include <iostream>


class ParallelB1 {
public:
void static PopulateTree(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
void static PopulateTreeParallel(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
int static PopulateTreeParallelLevelWise(int numberOfLevels, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
void static PopulateTreeParallelAsymmetric(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
int static GenerateChildren( Graph & child_g);
int static CalculateWorstCaseSpaceComplexity(int vertexCount);
long long static CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels);
void static TraverseUpTree(int index, 
                            std::vector<Graph> & graphs,
                            std::vector<int> & answer);


private:
    void static DFS(std::vector<int> & new_row_off,
                    std::vector<int> & new_col_ref,
                    std::vector<int> & new_vals_ref,
                    std::vector<int> & path, 
                    int rootVertex);
    int static GetRandomVertex(std::vector<int> & verticesRemaining);
    int static GetRandomOutgoingEdge(std::vector<int> & new_row_off,
                                    std::vector<int> & new_col_ref,
                                    std::vector<int> & new_values_ref,
                                    int v, 
                                    std::vector<int> & path);
    int static classifyPath(std::vector<int> & path);
    void static createVertexSetsForEachChild(std::vector< std::vector<int> > & childrensVertices, 
                                                int caseNumber, 
                                                std::vector<int> & path);
};

#endif
