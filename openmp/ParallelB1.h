
#ifndef Parallel_B1_H
#define Parallel_B1_H

#include "Graph.h"
#include "ParallelKernelization.h"
#include <iostream>


class ParallelB1 {
public:
    ParallelB1( Graph * g_arg,
                int k_arg,
                std::vector<int> & verticesToRemove_arg,
                //std::vector<int> verticesRemaining_arg,
                ParallelB1 * parent_arg = NULL);
    ~ParallelB1();
    void IterateTreeStructure(ParallelB1 * root);
    void TraverseUpTree(ParallelB1 * leaf, std::vector<int> & answer);
    int GetNumberChildren();
    std::vector<int> GetVerticesToRemove();
    ParallelB1 * GetChild(int i);
    ParallelB1 * GetParent();
    bool GetResult();

private:
    int classifyPath(std::vector<int> & path);
    void createVertexSetsForEachChild(int caseNumber, std::vector<int> & path);

    Graph * g;
    int k;
    std::vector<int> verticesToRemove;
    std::vector< std::vector<int> > childrensVertices;
    ParallelB1 * parent, ** children;
    bool result;

};

#endif
