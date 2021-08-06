
#ifndef Parallel_B1_H
#define Parallel_B1_H

#include "Graph.h"
#include "ParallelKernelization.h"
#include <iostream>


class ParallelB1 : public std::enable_shared_from_this<ParallelB1>{
public:
    ParallelB1( std::shared_ptr<Graph> g_arg,
                int k_arg,
                std::vector<int> & verticesToRemove_arg,
                //std::vector<int> verticesRemaining_arg,
                std::shared_ptr<ParallelB1> parent_arg = NULL);
    //~ParallelB1();
    bool IterateTreeStructure(  std::shared_ptr<ParallelB1> root,
                                std::vector<int> & answer);
    void TraverseUpTree(std::shared_ptr<ParallelB1> leaf, std::vector<int> & answer);
    int GetNumberChildren();
    std::vector<int> GetVerticesToRemove();
    std::shared_ptr<ParallelB1> GetChild(int i);
    std::shared_ptr<ParallelB1> GetParent();
    bool GetResult();

private:
    int classifyPath(std::vector<int> & path);
    void createVertexSetsForEachChild(int caseNumber, std::vector<int> & path);

    std::shared_ptr<Graph> g;

    int k;
    std::vector<int> verticesToRemove;
    std::vector< std::vector<int> > childrensVertices;
    std::shared_ptr<ParallelB1> parent;
    std::vector<std::shared_ptr<ParallelB1>> children;
    bool result;

};

#endif
