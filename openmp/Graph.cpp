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
    edgesLeftToCover = compressedSparseMatrix->column_indices.size()/2;
    verticesRemaining.resize(vertexCount);
    std::iota (std::begin(verticesRemaining), std::end(verticesRemaining), 0); // Fill with 0, 1, ..., 99.

}

/* Constructor to make induced subgraph G' post-kernelization */
Graph::Graph(Graph & g_arg)
{        
    /* This should use the edgesLeftToCover constructor of COO */
    compressedSparseMatrix = new CSR(*(g_arg.GetCSR()), g_arg.GetEdgesLeftToCover());             
    std::cout << compressedSparseMatrix->toString();
    edgesLeftToCover = compressedSparseMatrix->column_indices.size()/2;
}

/* Constructor to make induced subgraph G'' for each branch */
Graph::Graph(Graph & g_arg, std::vector<int> & verticesToDelete)
{        
    std::cout << "Called the delete verts constructor" << std::endl;
    compressedSparseMatrix = new CSR(*(g_arg.GetCSR()), verticesToDelete);
    std::cout << "Built the CSR" << std::endl;
    std::cout << compressedSparseMatrix->toString();
    edgesLeftToCover = compressedSparseMatrix->column_indices.size()/2;
}

/* Constructor to make induced subgraph G'' for each branch */
// Called the CSR delete verts constructor in the method call

Graph::Graph(CSR * csr_arg, std::vector<int> & verticesToDelete):
    compressedSparseMatrix(csr_arg)
{        
    // Sets some of the entries in values[] to 0
    SetEdgesOfSSymParallel(verticesToDelete); 
    std::cout << compressedSparseMatrix->toString();
    edgesLeftToCover = compressedSparseMatrix->column_indices.size()/2;
}

std::vector<int> & Graph::GetRemainingVertices(){
    return verticesRemaining;
}


int Graph::GetRandomVertex(){
    //return degCont->GetRandomVertex();
        return 1;

}


COO * Graph::GetCOO(){
    return coordinateFormat;
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
        /* To prevent simple paths, must at least have 2 entries, 
        assuming there are no self edges, since the first entry, v,
        is randomly chosen and the second entry is a random out edge */
        if (path.size() > 1 && *it == path.rbegin()[1]) {
            //std::cout << "Wouldve been a simple path, skipping " << *it << std::endl;
            ++it;
        } else
            return *it;
    }

    return -1;
}


/* The edges */
CSR * Graph::GetCSR(){
    return compressedSparseMatrix;
}

void Graph::removeVertex(int vertexToRemove, std::vector<int> & verticesRemaining){
        std::vector<int>::iterator low;
        low = std::lower_bound( std::begin(verticesRemaining), 
                                std::end(verticesRemaining), 
                                vertexToRemove);
        if (low == std::end(verticesRemaining))
            return;
        else
            verticesRemaining.erase(low);
        
}

void Graph::SetEdgesOfSSymParallel(std::vector<int> & S){
    int v, intraRowOffset;
    std::vector<int>::iterator low;
    std::vector<int> & row_offsets_ref = compressedSparseMatrix->GetOldRowOffRef();
    std::vector<int> & column_indices_ref = compressedSparseMatrix->GetOldColRef();
    std::vector<int> & values = compressedSparseMatrix->GetNewValRef();

    // Set out-edges
    #pragma omp parallel for default(none) shared(row_offsets_ref, \
    column_indices_ref, values, S) private (v)
    for (auto u : S)
    {
        for (int i = row_offsets_ref[u]; i < row_offsets_ref[u+1]; ++i){
            v = column_indices_ref[i];
            values[i] = 0;
        }
    }

    // Set in-edges
    #pragma omp parallel for default(none) shared(row_offsets_ref, \
    column_indices_ref, values, S) private (low, v, intraRowOffset)
    for (auto u : S)
    {
        for (int i = row_offsets_ref[u]; i < row_offsets_ref[u+1]; ++i){
            v = column_indices_ref[i];
            /* Break this into 2 independent for loops */
            //!!!!!   a must be sorted by cols within rows.       
            low = std::lower_bound( column_indices_ref.begin() + row_offsets_ref[v], 
                                    column_indices_ref.begin() + row_offsets_ref[v+1], 
                                    u);
            intraRowOffset = low - (column_indices_ref.begin() + row_offsets_ref[v]);
            // Set in-edge
            values[row_offsets_ref[v] + intraRowOffset] = 0;
        }
    }
}
/*

void Graph::SetEdgesLeftToCoverParallel(){
    int count = 0, i = 0, j = 0;
    std::vector<int> & newDegs = newDegrees;
    std::vector<int> & row_offsets = compressedSparseMatrix->GetOldRowOffSets();
    #pragma omp parallel for default(none) shared(row_offsets, values, newDegs) private (i, j) \
    reduction(+:count)
    for (i = 0; i < numberOfRows; ++i)
    {
        for (j = row_offsets[i]; j < row_offsets[i+1]; ++j)
            newDegs[i] += values[j];
        count += newDegs[i];
    }
    g.edgesLeftToCover = count;
}

void Graph::SetNewRowOffsets(){
    int i = 0;
    std::vector<int> & newDegs = newDegrees;
    std::vector<int> & newRowOffs = compressedSparseMatrix->GetRowOffSets();
    newRowOffs.resize(numberOfRows+1);
    for (i = 1; i <= numberOfRows; ++i)
    {
        newRowOffs[i] = newDegs[i-1] + newRowOffs[i-1];
    }
}

*/