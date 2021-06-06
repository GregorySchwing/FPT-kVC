#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <vector>
#include "COO.h"
#include "CSR.h"
#include "DegreeController.h"
#include "NeighborsBinaryDataStructure.h"

class Graph {
    public:
        Graph(int vertexCount);
        DegreeController * GetDegreeController();
        int GetDegree(int v);
        int edgesLeftToCover;

    private:
        COO coordinateFormat;
        CSR * compressedSparseMatrix;
        DegreeController * degCont;
        NeighborsBinaryDataStructure * neighBits;
};
#endif
