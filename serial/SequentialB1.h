
#ifndef Sequential_B1_H
#define Sequential_B1_H

#include "Graph.h"
#include <iostream>

class SequentialB1 {
public:
    SequentialB1(Graph & g_arg, int k_arg, int k_prime_arg);
private:
    /* Indicates whether at a given node the vertex v or the neighbours of v
    are included in the cover */
    bool * boundedSearchTreeChoiceIndicator; 
    int * verticesEdgesCovered; 
    Graph & g;
    int k;
    int k_prime;
};

#endif
