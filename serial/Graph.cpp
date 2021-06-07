#include "Graph.h"

Graph::Graph(int vertexCount): coordinateFormat(vertexCount, vertexCount)
{
    /* Eventually replace this with an initialization from file */
    coordinateFormat.addEdgeASymmetric(2,0,5);
    coordinateFormat.addEdgeASymmetric(1,3,2);
    coordinateFormat.addEdgeASymmetric(2,1,2);
    coordinateFormat.addEdgeASymmetric(2,3,2);
    coordinateFormat.addEdgeASymmetric(2,4,2);
    coordinateFormat.addEdgeASymmetric(1,4,2);
    coordinateFormat.addEdgeASymmetric(0,4,2);

    coordinateFormat.size = coordinateFormat.column_indices.size();
    // vlog(e)
    coordinateFormat.sortMyself();
    compressedSparseMatrix = new CSR(coordinateFormat);             
    std::cout << coordinateFormat.toString();
    std::cout << compressedSparseMatrix->toString();
    neighBits = new NeighborsBinaryDataStructure(compressedSparseMatrix);
    degCont = new DegreeController(compressedSparseMatrix->numberOfRows, neighBits);
    std::cout << degCont->toString();
    edgesLeftToCover = compressedSparseMatrix->column_indices.size();
}

DegreeController * Graph::GetDegreeController(){
    return degCont;
}

int Graph::GetDegree(int v){
    return compressedSparseMatrix->row_offsets[v+1] - compressedSparseMatrix->row_offsets[v];
}
