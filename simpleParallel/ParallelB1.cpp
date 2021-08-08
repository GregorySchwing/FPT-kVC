#include "ParallelB1.h"

void ParallelB1::InduceSubgraph( Graph & child_g,
                                Graph & parent_g){
    
}

int ParallelB1::classifyPath(std::vector<int> & path){
    if (path.size()==2)
        return 3;
    else if (path.size()==3)
        return 2;
    else if (path.front() == path.back())
        return 1;
    else
        return 0;
}

void ParallelB1::createVertexSetsForEachChild(std::vector< std::vector<int> > & childrensVertices,
                                            int caseNumber, 
                                            std::vector<int> & path){
    if (caseNumber == 0) {
        /* 3 Children */
        childrensVertices.resize(3);
        /* Each with 2 vertices */
        for (auto & cV : childrensVertices)
            cV.reserve(2);
        childrensVertices[0].push_back(path[0]);
        childrensVertices[0].push_back(path[2]);

        childrensVertices[1].push_back(path[1]);
        childrensVertices[1].push_back(path[2]);

        childrensVertices[2].push_back(path[1]);
        childrensVertices[2].push_back(path[3]);

    } else if (caseNumber == 1) {

        /* 3 Children */
        childrensVertices.resize(3);
        /* Each with 2 vertices */
        for (auto & cV : childrensVertices)
            cV.reserve(2);
        childrensVertices[0].push_back(path[0]);
        childrensVertices[0].push_back(path[1]);

        childrensVertices[1].push_back(path[1]);
        childrensVertices[1].push_back(path[2]);

        childrensVertices[2].push_back(path[0]);
        childrensVertices[2].push_back(path[2]);

    } else if (caseNumber == 2) {

        childrensVertices.resize(1);
        childrensVertices[0].reserve(1);
        childrensVertices[0].push_back(path[1]);

    } else {

        childrensVertices.resize(1);
        childrensVertices[0].reserve(1);
        childrensVertices[0].push_back(path[0]);

    }
}

