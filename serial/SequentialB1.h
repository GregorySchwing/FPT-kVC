
#ifndef Sequential_B1_H
#define Sequential_B1_H

#include "Graph.h"
#include "SequentialKernelization.h"
#include <iostream>


class SequentialB1 {
public:
    SequentialB1(   Graph * g_arg, 
                    int k_prime_arg,
                    SequentialB1 * parent_arg = NULL);
private:
    /* DFS of maximum length 3. No simple cycles u -> v -> u */
    void DFS(std::vector<int> & path, int rootVertex);
    int classifyPath(std::vector<int> & path);
    void createVertexSetsForEachChild(int caseNumber, std::vector<int> & path);
    Graph * g;
    int k_prime;
    std::vector< std::vector<int> > childrensVertices;
    SequentialB1 * parent, ** children;

};

#endif
