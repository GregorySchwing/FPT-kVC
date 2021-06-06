#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <vector>
#include "COO.h"
#include "CSR.h"
#include "DegreeController.h"
class Graph {
    public:
        Graph(int vertexCount);
        DegreeController * GetDegreeController();
    private:
        COO coordinateFormat;
        CSR * compressedSparseMatrix;
        DegreeController * degCont;
};
#endif
