#include "Graph.h"

Graph::Graph(int vertexCount)
{
    coordinateFormat = new COO(vertexCount, vertexCount);

    /* Eventually replace this with an initialization from file */
    coordinateFormat->addEdgeSymmetric(0,1,1);
    coordinateFormat->addEdgeSymmetric(0,4,1);
    coordinateFormat->addEdgeSymmetric(1,4,1);
    coordinateFormat->addEdgeSymmetric(1,5,1);
    coordinateFormat->addEdgeSymmetric(1,6,1);
    coordinateFormat->addEdgeSymmetric(2,4,1);
    coordinateFormat->addEdgeSymmetric(2,6,1);
    coordinateFormat->addEdgeSymmetric(3,5,1);
    coordinateFormat->addEdgeSymmetric(3,6,1);
    coordinateFormat->addEdgeSymmetric(4,7,1);
    coordinateFormat->addEdgeSymmetric(4,8,1);
    coordinateFormat->addEdgeSymmetric(5,8,1);
    coordinateFormat->addEdgeSymmetric(6,9,1);

    coordinateFormat->size = coordinateFormat->column_indices.size();
    // vlog(e)
    coordinateFormat->sortMyself();
    compressedSparseMatrix = new CSR(*coordinateFormat);             
    std::cout << coordinateFormat->toString();
    std::cout << compressedSparseMatrix->toString();
    neighBits = new NeighborsBinaryDataStructure(compressedSparseMatrix);
    degCont = new DegreeController(compressedSparseMatrix->numberOfRows, neighBits);
    std::cout << degCont->toString();
    edgesLeftToCover = compressedSparseMatrix->column_indices.size()/2;
}

/* Constructor to make induced subgraph G' */
Graph::Graph(Graph & g_arg)
{        
    /* This should use the edgesLeftToCover constructor of COO */
    compressedSparseMatrix = new CSR(*(g_arg.GetCSR()), g_arg.GetEdgesLeftToCover());             
    std::cout << compressedSparseMatrix->toString();
    neighBits = new NeighborsBinaryDataStructure(compressedSparseMatrix);
    degCont = new DegreeController(compressedSparseMatrix->numberOfRows, neighBits);
    std::cout << degCont->toString();
    edgesLeftToCover = compressedSparseMatrix->column_indices.size()/2;
    
}

void Graph::UpdateNeighBits(){

}

COO * Graph::GetCOO(){
    return coordinateFormat;
}


DegreeController * Graph::GetDegreeController(){
    return degCont;
}

int Graph::GetEdgesLeftToCover(){
    return edgesLeftToCover;
}


int Graph::GetDegree(int v){
    return compressedSparseMatrix->row_offsets[v+1] - compressedSparseMatrix->row_offsets[v];
}

/* The edges */
CSR * Graph::GetCSR(){
    return compressedSparseMatrix;
}


