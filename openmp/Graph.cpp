#include "Graph.h"

Graph::Graph(int vertexCount): 
// Circular reference since there are no old degrees.
old_degrees_ref(new_degrees)
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

/* Constructor to make induced subgraph G'' for each branch */
// Called the CSR delete verts constructor in the method call

Graph::Graph(Graph * g_arg, std::vector<int> & verticesToDelete):
    compressedSparseMatrix(g_arg->GetCSR()),
    // Will change the CSR arg to a G so the old deg ref is available.
    old_degrees_ref(g_arg->new_degrees)
{        
    // Sets some of the entries in values[] to 0
    SetEdgesOfSSymParallel(verticesToDelete); 
    // Get an accurate number of edges left so we can 
    // decide if we are done before starting next branch
    SetEdgesLeftToCoverParallel();
    std::cout << compressedSparseMatrix->toString();
    std::cout << edgesLeftToCover << " edges left in induced subgraph G'" << std::endl;
}

std::vector<int> & Graph::GetRemainingVerticesRef(){
    return verticesRemaining;
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

int Graph::GetNumberOfRows(){
    return numberOfRows;
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


void Graph::SetEdgesLeftToCoverParallel(){
    int count = 0, i = 0, j = 0;
    std::vector<int> & newDegs = new_degrees;
    std::vector<int> & values = compressedSparseMatrix->GetNewValRef();
    std::vector<int> & row_offsets = compressedSparseMatrix->GetOldRowOffRef();
    #pragma omp parallel for default(none) shared(row_offsets, values, newDegs, numberOfRows) private (i, j) \
    reduction(+:count)
    for (i = 0; i < numberOfRows; ++i)
    {
        for (j = row_offsets[i]; j < row_offsets[i+1]; ++j)
            newDegs[i] += values[j];
        count += newDegs[i];
    }
    edgesLeftToCover = count;
}

/* Called if edgesLeftToCover > 0 */
void Graph::SetNewRowOffsets(std::vector<int> & newRowOffsetsRef){
    int i = 0;
    newRowOffsetsRef.resize(numberOfRows+1);
    for (i = 1; i <= numberOfRows; ++i)
    {
        newRowOffsetsRef[i] = new_degrees[i-1] + newRowOffsetsRef[i-1];
    }
}


void Graph::CountingSortParallelRowwiseValues(
                int rowID,
                int beginIndex,
                int endIndex,
                std::vector<int> & A_row_offsets,
                std::vector<int> & A_column_indices,
                std::vector<int> & A_values,
                std::vector<int> & B_row_indices_ref,
                std::vector<int> & B_column_indices_ref,
                std::vector<int> & B_values_ref){

    //std::cout << "procID : " << procID << " beginIndex " << beginIndex << " endIndex " << endIndex << std::endl;

    int max = 1;

    std::vector<int> C_ref(max+1, 0);

    for (int i = beginIndex; i < endIndex; ++i){
        ++C_ref[A_values[i]];
    }

    //std::cout << "C[i] now contains the number elements equal to i." << std::endl;
    for (int i = 1; i < max+1; ++i){
        C_ref[i] = C_ref[i] + C_ref[i-1];
    }
    /* C_ref[A_row_indices[i]]]-1 , because the values of C_ref are from [1, n] -> [0,n) */
    for (int i = endIndex-1; i >= beginIndex; --i){
        if (A_values[i]){
            std::cout << (B_row_indices_ref[rowID] - C_ref[0] + C_ref[1] -1) << std::endl;
            B_column_indices_ref[B_row_indices_ref[rowID] - C_ref[0] + C_ref[1]-1] = A_column_indices[i];
            B_values_ref[B_row_indices_ref[rowID] - C_ref[0] + C_ref[1]-1] = A_values[i];
            --C_ref[A_values[i]];
        }
    }
}

void Graph::RemoveDegreeZeroVertices(std::vector<int> & newRowOffsets){
    int i = 0;
    for (i = 0; i < numberOfRows; ++i){
        if(newRowOffsets[i+1] - newRowOffsets[i] == 0)
            removeVertex(i, verticesRemaining);
    }
}

void Graph::PrepareGPrime(){
        
        compressedSparseMatrix->GetNewColRef().resize(edgesLeftToCover);
        // Temporary, used for checking, will be replaced by vector of all 1's


        std::vector<int> & row_offsets = compressedSparseMatrix->GetOldRowOffRef();
        std::vector<int> & column_indices = compressedSparseMatrix->GetOldColRef();
        std::vector<int> & values = compressedSparseMatrix->GetNewValRef();
        
        std::vector<int> & newRowOffsets = compressedSparseMatrix->GetNewColRef();
        std::vector<int> & newColumnIndices = compressedSparseMatrix->GetNewColRef();
        //std::vector<int> & newValuesRef;

        newValues.resize(edgesLeftToCover);
        SetNewRowOffsets(newRowOffsets);
        int row; 
        
        #pragma omp parallel for default(none) \
                            shared(row_offsets, column_indices, values, \
                            new_degrees, newRowOffsets, newColumnIndices, newValues) \
                            private (row)
        for (row = 0; row < numberOfRows; ++row)
        {
            CountingSortParallelRowwiseValues(row,
                                            row_offsets[row],
                                            row_offsets[row+1],
                                            row_offsets,
                                            column_indices,
                                            values,
                                            newRowOffsets,
                                            newColumnIndices,
                                            newValues);
        }

        RemoveDegreeZeroVertices(newRowOffsets);
}

std::vector<int> & Graph::GetCondensedNewValRef(){
    return newValues;
}

/* DFS of maximum length 3. No simple cycles u -> v -> u */
void Graph::DFS(std::vector<int> & path, int rootVertex){
    if (path.size() == 4)
        return;

    int randomOutgoingEdge = GetRandomOutgoingEdge(rootVertex, path);
    if (randomOutgoingEdge < 0) {
        return;
    } else {
        path.push_back(randomOutgoingEdge);
        return DFS(path, randomOutgoingEdge);
    }
}