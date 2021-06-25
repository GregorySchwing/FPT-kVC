
#ifndef Sequential_B1_H
#define Sequential_B1_H

#include "Graph.h"
#include "SequentialKernelization.h"
#include <iostream>


class SequentialB1 {
public:
    SequentialB1(   Graph * g_arg, 
                    std::vector<int> verticesToRemove,
                    int k_prime_arg,
                    SequentialB1 * parent_arg = NULL);

    void IterateTreeStructure(SequentialB1 * root);
    void TraverseUpTree(SequentialB1 * leaf, std::vector<int> & answer);
    int GetNumberChildren();
    std::vector<int> GetVerticesToRemove();
    SequentialB1 * GetChild(int i);
    SequentialB1 * GetParent();
    bool GetResult();


private:
    /* DFS of maximum length 3. No simple cycles u -> v -> u */
    void DFS(std::vector<int> & path, int rootVertex);
    int classifyPath(std::vector<int> & path);
    void createVertexSetsForEachChild(int caseNumber, std::vector<int> & path);
    Graph * g;
    int k_prime;
    std::vector<int> verticesToRemove;
    std::vector< std::vector<int> > childrensVertices;
    SequentialB1 * parent, ** children;
    bool result;

};

#endif
