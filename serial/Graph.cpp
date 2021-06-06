#include "Graph.h"

Graph::Graph(int vertexCount): coordinateFormat(vertexCount, vertexCount)
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

DegreeController * Graph::GetDegreeController(){
    return degCont;
}