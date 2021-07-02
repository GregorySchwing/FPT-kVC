
#ifndef Parallel_B1_H
#define Parallel_B1_H

#include "Graph.h"
#include "ParallelKernelization.h"
#include <iostream>


class ParallelB1 {
public:
    ParallelB1( int k_prime_arg,
                ParallelKernelization * pk_arg,
                ParallelB1 * parent_arg = NULL);

    ParallelB1( Graph * g_arg,
                int k_prime_arg,
                ParallelB1 * parent_arg,
                std::vector<int> & verticesToRemove,
                std::vector<int> verticesRemaining);
/*
    ParallelB1(   Graph * g_arg, 
                    std::vector<int> verticesToRemove,
                    int k_prime_arg,
                    ParallelB1 * parent_arg = NULL);



    void IterateTreeStructure(ParallelB1 * root);
    void TraverseUpTree(ParallelB1 * leaf, std::vector<int> & answer);
    int GetNumberChildren();
    std::vector<int> GetVerticesToRemove();
    ParallelB1 * GetChild(int i);
    ParallelB1 * GetParent();
    bool GetResult();
*/

private:
    /* DFS of maximum length 3. No simple cycles u -> v -> u */
    void DFS(std::vector<int> & path, int rootVertex);  
    int classifyPath(std::vector<int> & path);
    void createVertexSetsForEachChild(int caseNumber, std::vector<int> & path);

    Graph * g;
    ParallelKernelization * pk;
    int k_prime;
    std::vector<int> verticesToRemove;
    std::vector< std::vector<int> > childrensVertices;
    ParallelB1 * parent, ** children;
    bool result;

    std::vector<int> newDegrees, newRowOffsets, newColumnIndices, newValues, vertexTouchedByRemovedEdge;
};

#endif
