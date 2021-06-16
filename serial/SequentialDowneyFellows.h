
#ifndef Sequential_Downey_Fellows_H
#define Sequential_Downey_Fellows_H

#include "Graph.h"
#include <iostream>

class SequentialDowneyFellows {
public:
    SequentialDowneyFellows();
private:
    /* Indicates whether at a given node the vertex v or the neighbours of v
    are included in the cover */
    bool * boundedSearchTreeChoiceIndicator; 
    int * verticesEdgesCovered; 
};

#endif
