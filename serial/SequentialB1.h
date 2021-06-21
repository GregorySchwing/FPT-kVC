
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
    void DFS(std::vector<int> & path, int rootVertex);
    SequentialB1 * parent;
    Graph & g;
    int k_prime;
};

#endif
