
#ifndef Parallel_B1_H
#define Parallel_B1_H

#include "Graph.h"
#include "ParallelKernelization.h"
#include <iostream>


class ParallelB1 {
public:
void static InduceSubgraph( Graph & child_g,
                            Graph & parent_g);

private:
    void static DFS(std::vector<int> & new_row_off,
                    std::vector<int> & new_col_ref,
                    std::vector<int> & path, 
                    int rootVertex);
    int static GetRandomVertex(std::vector<int> & verticesRemaining);
    int static GetRandomOutgoingEdge(std::vector<int> & new_row_off,
                                    std::vector<int> & new_col_ref,
                                    int v, 
                                    std::vector<int> & path);
    int static classifyPath(std::vector<int> & path);
    void static createVertexSetsForEachChild(std::vector< std::vector<int> > & childrensVertices, 
                                                int caseNumber, 
                                                std::vector<int> & path);
};

#endif
