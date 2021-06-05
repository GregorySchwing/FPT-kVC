#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <vector>
#include "COO.h"
#include "CSR.h"
#include "DegreeController.h"
class Graph {
    public:
        Graph(int vertexCount): coordinateFormat(vertexCount, vertexCount)
        {
            /* Eventually replace this with an initialization from file */
            coordinateFormat.addEdge(2,0,5);
            coordinateFormat.addEdge(1,3,2);
            coordinateFormat.addEdge(0,1,2);
            coordinateFormat.addEdge(0,3,2);
            coordinateFormat.size = coordinateFormat.column_indices.size();
            coordinateFormat.sortMyself();
            compressedSparseMatrix = new CSR(coordinateFormat);             
            std::cout << coordinateFormat.toString();
            std::cout << compressedSparseMatrix->toString();
            //degCont = new DegreeController(compressedSparseMatrix);
            degCont = new DegreeController(compressedSparseMatrix);
            std::cout << degCont->toString();
        }

        DegreeController * GetDegreeController(){
            return degCont;
        }
    private:
        COO coordinateFormat;
        CSR * compressedSparseMatrix;
        DegreeController * degCont;
};
#endif
