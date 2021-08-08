
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
    int static classifyPath(std::vector<int> & path);
    void static createVertexSetsForEachChild(std::vector< std::vector<int> > & childrensVertices, 
                                                int caseNumber, 
                                                std::vector<int> & path);
};

#endif
