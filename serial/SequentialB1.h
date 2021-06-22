
#ifndef Sequential_B1_H
#define Sequential_B1_H

#include "Graph.h"
#include "SequentialKernelization.h"
#include <iostream>


class SequentialB1 {
public:
    SequentialB1(   Graph & g_arg, 
                    int k_prime_arg);
private:
    /* DFS of maximum length 3. No simple cycles u -> v -> u */
    void DFS(std::vector<int> & path, int rootVertex);
    int classifyPath(std::vector<int> & path);
    SequentialB1 * parent;
    Graph & g;
    int k_prime;
};

#endif
