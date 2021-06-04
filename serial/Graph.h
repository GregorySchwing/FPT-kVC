#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <vector>
#include "COO.h"
#include "CSR.h"
class Graph {
    public:
        Graph(int vertexCount): coordinateFormat(vertexCount, vertexCount), compressedSparseMatrix(coordinateFormat)
        {
                        
        }
    private:
        COO coordinateFormat;
        CSR compressedSparseMatrix;
};
#endif
