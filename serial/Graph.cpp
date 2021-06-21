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
    /* If we use the Asymetric */
    //degCont = new DegreeController(compressedSparseMatrix->numberOfRows, neighBits);
    /* If we use the Symetric */
    degCont = new DegreeController(compressedSparseMatrix->numberOfRows, compressedSparseMatrix);
    
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
    /* If we use the Asymetric */
    //degCont = new DegreeController(compressedSparseMatrix->numberOfRows, neighBits);
    /* If we use the Symetric */
    degCont = new DegreeController(compressedSparseMatrix->numberOfRows, compressedSparseMatrix);
    std::cout << degCont->toString();
    edgesLeftToCover = compressedSparseMatrix->column_indices.size()/2;
}

void Graph::UpdateNeighBits(){

}

int Graph::GetRandomVertex(){
    return degCont->GetRandomVertex();
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

int Graph::GetOutgoingEdge(int v, int outEdgeIndex){
    return compressedSparseMatrix->column_indices[compressedSparseMatrix->row_offsets[v] + outEdgeIndex];
}

int Graph::GetRandomOutgoingEdge(int v, std::vector<int> & path){

    std::vector<int> outgoingEdges(&compressedSparseMatrix->column_indices[compressedSparseMatrix->row_offsets[v]],
                        &compressedSparseMatrix->column_indices[compressedSparseMatrix->row_offsets[v+1]]);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(outgoingEdges.begin(), outgoingEdges.end(), g);
    std::vector<int>::iterator it = outgoingEdges.begin();
    
    while (it != outgoingEdges.end()){
        if (*it == path.back())
            ++it;
        else
            return *it;
    }

    return -1;
}


/* The edges */
CSR * Graph::GetCSR(){
    return compressedSparseMatrix;
}


