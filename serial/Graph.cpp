#include "Graph.h"

Graph::Graph(int vertexCount): coordinateFormat(vertexCount, vertexCount)
{
    /* Eventually replace this with an initialization from file */
    coordinateFormat.addEdgeSymmetric(2,0,5);
    coordinateFormat.addEdgeSymmetric(1,3,2);
    coordinateFormat.addEdgeSymmetric(2,1,2);
    coordinateFormat.addEdgeSymmetric(2,3,2);
    coordinateFormat.size = coordinateFormat.column_indices.size();
    coordinateFormat.sortMyself();
    compressedSparseMatrix = new CSR(coordinateFormat);             
    std::cout << coordinateFormat.toString();
    std::cout << compressedSparseMatrix->toString();
    degCont = new DegreeController(compressedSparseMatrix);
    std::cout << degCont->toString();
    neighBits = new NeighborsBinaryDataStructure(compressedSparseMatrix);
}

DegreeController * Graph::GetDegreeController(){
    return degCont;
}