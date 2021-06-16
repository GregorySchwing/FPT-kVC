#include "Graph.h"

Graph::Graph(int vertexCount): coordinateFormat(vertexCount, vertexCount)
{
    /* Eventually replace this with an initialization from file */
    coordinateFormat.addEdgeSymmetric(0,1,1);
    coordinateFormat.addEdgeSymmetric(0,4,1);
    coordinateFormat.addEdgeSymmetric(1,4,1);
    coordinateFormat.addEdgeSymmetric(1,5,1);
    coordinateFormat.addEdgeSymmetric(1,6,1);
    coordinateFormat.addEdgeSymmetric(2,4,1);
    coordinateFormat.addEdgeSymmetric(2,6,1);
    coordinateFormat.addEdgeSymmetric(3,5,1);
    coordinateFormat.addEdgeSymmetric(3,6,1);
    coordinateFormat.addEdgeSymmetric(4,7,1);
    coordinateFormat.addEdgeSymmetric(4,8,1);
    coordinateFormat.addEdgeSymmetric(5,8,1);
    coordinateFormat.addEdgeSymmetric(6,9,1);

    coordinateFormat.size = coordinateFormat.column_indices.size();
    // vlog(e)
    coordinateFormat.sortMyself();
    compressedSparseMatrix = new CSR(coordinateFormat);             
    std::cout << coordinateFormat.toString();
    std::cout << compressedSparseMatrix->toString();
    neighBits = new NeighborsBinaryDataStructure(compressedSparseMatrix);
    degCont = new DegreeController(compressedSparseMatrix->numberOfRows, neighBits);
    std::cout << degCont->toString();
    edgesLeftToCover = compressedSparseMatrix->column_indices.size()/2;
}

/* Constructor to make induced subgraph G' */
Graph::Graph(std::vector<int> S, Graph & g_arg): coordinateFormat(g_arg.GetCOO())
{
    coordinateFormat.size = coordinateFormat.column_indices.size();
    if (!coordinateFormat.getIsSorted())
        // vlog(e)
        coordinateFormat.sortMyself();
    compressedSparseMatrix = new CSR(coordinateFormat);             
    std::cout << coordinateFormat.toString();
    std::cout << compressedSparseMatrix->toString();
    neighBits = new NeighborsBinaryDataStructure(compressedSparseMatrix);
    degCont = new DegreeController(compressedSparseMatrix->numberOfRows, neighBits);
    std::cout << degCont->toString();
    edgesLeftToCover = compressedSparseMatrix->column_indices.size()/2;
}

void Graph::UpdateNeighBits(){

}

COO & Graph::GetCOO(){
    return coordinateFormat;
}


DegreeController * Graph::GetDegreeController(){
    return degCont;
}

int Graph::GetDegree(int v){
    return compressedSparseMatrix->row_offsets[v+1] - compressedSparseMatrix->row_offsets[v];
}

/* The edges */
CSR * Graph::GetCSR(){
    return compressedSparseMatrix;
}


