#include "Graph.h"

Graph::Graph(int vertexCount): 
// Circular reference since there are no old degrees.
old_degrees_ref(new_degrees),
vertexCount(vertexCount)
{
    coordinateFormat = new COO(vertexCount, vertexCount);

    /* Eventually replace this with an initialization from file */
    BuildTheExampleCOO(coordinateFormat);
    ProcessGraph(vertexCount);
}

Graph::Graph(std::string filename, char sep, int vertexCount):
// Circular reference since there are no old degrees.
old_degrees_ref(new_degrees)
{
    coordinateFormat = new COO();
    /* Eventually replace this with an initialization from file */
    BuildCOOFromFile(coordinateFormat, filename);
    if(vertexCount == 0)
        SetVertexCountFromEdges(coordinateFormat);
    ProcessGraph(coordinateFormat->GetNumberOfRows());
}


/* Constructor to make induced subgraph G'' for each branch */
Graph::Graph(Graph * g_arg, std::vector<int> & verticesToDelete):
    // Sets the old references of the new csr 
    // to point to the new references of the argument
    compressedSparseMatrix(new CSR(g_arg->GetCSR())),
    old_degrees_ref(g_arg->new_degrees),
    vertexCount(g_arg->vertexCount)
{        
    std::cout << "Entered constructor of G induced" << std::endl;
    verticesRemaining = g_arg->verticesRemaining;
    new_degrees.resize(vertexCount);
    std::cout << compressedSparseMatrix->toString();

    // Sets some of the entries in values[] to 0
    SetEdgesOfSSymParallel(verticesToDelete); 
    // Get an accurate number of edges left so we can 
    // decide if we are done before starting next branch
    SetEdgesLeftToCoverParallel();
    std::cout << compressedSparseMatrix->toString();
    std::cout << "new vals" << std::endl;
    for (auto & v : compressedSparseMatrix->new_values)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << edgesLeftToCover << " edges left in induced subgraph G'" << std::endl;
}

void Graph::ProcessGraph(int vertexCount){
    compressedSparseMatrix = new CSR(*coordinateFormat);             
    std::cout << coordinateFormat->toString();
    std::cout << compressedSparseMatrix->toString();
    edgesLeftToCover = compressedSparseMatrix->new_column_indices.size()/2;
    verticesRemaining.resize(vertexCount);
    std::iota (std::begin(verticesRemaining), std::end(verticesRemaining), 0); // Fill with 0, 1, ..., 99.
    std::vector<int> & new_row_offsets = compressedSparseMatrix->new_row_offsets;
    new_degrees.resize(vertexCount);
    for (int i = 0; i < vertexCount; ++i){
        new_degrees[i] = new_row_offsets[i+1] - new_row_offsets[i];
    }
}

void Graph::SetVertexCountFromEdges(COO * coordinateFormat){
    int min;
    auto it = min_element(std::begin(coordinateFormat->new_row_indices), std::end(coordinateFormat->new_row_indices)); // C++11
    min = *it;
    it = max_element(std::begin(coordinateFormat->new_column_indices), std::end(coordinateFormat->new_column_indices)); // C++11
    if(min > *it)
        min = *it;
    if(min != 0){
        int scaleToRenumberAtZero = 0 - min;
        for (auto & v : coordinateFormat->new_row_indices)
            v += scaleToRenumberAtZero;
        for (auto & v : coordinateFormat->new_column_indices)
            v += scaleToRenumberAtZero;
    }
            
    int max;
    it = max_element(std::begin(coordinateFormat->new_row_indices), std::end(coordinateFormat->new_row_indices)); // C++11
    max = *it;
    it = max_element(std::begin(coordinateFormat->new_column_indices), std::end(coordinateFormat->new_column_indices)); // C++11
    if(max < *it)
        max = *it;
    coordinateFormat->SetNumberOfRows(max);
    this->vertexCount = max;

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
    return compressedSparseMatrix->old_row_offsets_ref[v+1] - compressedSparseMatrix->old_row_offsets_ref[v];
}

int Graph::GetOutgoingEdge(int v, int outEdgeIndex){
    return compressedSparseMatrix->old_column_indices_ref[compressedSparseMatrix->old_row_offsets_ref[v] + outEdgeIndex];
}

int Graph::GetVertexCount(){
    return vertexCount;
}


int Graph::GetRandomOutgoingEdge(int v, std::vector<int> & path){

    std::vector<int> outgoingEdges(&compressedSparseMatrix->new_column_indices[compressedSparseMatrix->new_row_offsets[v]],
                        &compressedSparseMatrix->new_column_indices[compressedSparseMatrix->new_row_offsets[v+1]]);

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
CSR & Graph::GetCSR(){
    return *compressedSparseMatrix;
}

void Graph::removeVertex(int vertexToRemove, std::vector<int> & verticesRemaining){
        std::vector<int>::iterator low;
        low = std::lower_bound( std::begin(verticesRemaining), 
                                std::end(verticesRemaining), 
                                vertexToRemove);
        if (low == std::end(verticesRemaining))
            return;
        else{
            std::cout << "Begginning of a call" << std::endl;
            std::cout << "vertexToRemove " << vertexToRemove << std::endl;

            std::cout << "Before Erasing " << low - std::begin(verticesRemaining) << std::endl;
            for ( auto & v : verticesRemaining )
                std::cout << v << " ";
            std::cout << std::endl;

            std::cout << "Erasing " << low - std::begin(verticesRemaining) << std::endl;
            verticesRemaining.erase(low);
            std::cout << "After Erasing " << low - std::begin(verticesRemaining) << std::endl;
            for ( auto & v : verticesRemaining )
                std::cout << v << " ";
            std::cout << std::endl;
            std::cout << "End of a call" << std::endl;
        }
        
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
    #pragma omp parallel for default(none) shared(row_offsets, values, newDegs, vertexCount) private (i, j) \
    reduction(+:count)
    for (i = 0; i < vertexCount; ++i)
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
    newRowOffsetsRef.resize(vertexCount+1);
    for (i = 1; i <= vertexCount; ++i)
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
            B_column_indices_ref[B_row_indices_ref[rowID] - C_ref[0] + C_ref[1]-1] = A_column_indices[i];
            B_values_ref[B_row_indices_ref[rowID] - C_ref[0] + C_ref[1]-1] = A_values[i];
            --C_ref[A_values[i]];
        }
    }
}

// Highly unoptimized, but should work for now
void Graph::RemoveNewlyDegreeZeroVertices(  std::vector<int> & verticesToRemove,
                                            std::vector<int> & oldRowOffsets, 
                                            std::vector<int> & oldColumnIndices, 
                                            std::vector<int> & newRowOffsets){
    int i = 0, j;
    std::vector<int> hasntBeenRemoved(vertexCount, 1);
    for (auto & v :verticesToRemove){
        removeVertex(v, verticesRemaining);
        hasntBeenRemoved[v] = 0;
        for (i = oldRowOffsets[v]; i < oldRowOffsets[v+1]; ++i){
            j = oldColumnIndices[i];
            if(newRowOffsets[j+1] - newRowOffsets[j] == 0)
                if (hasntBeenRemoved[j]){
                    removeVertex(j, verticesRemaining);
                    hasntBeenRemoved[j] = 0;
                }
        }
    }
}

void Graph::PrepareGPrime(std::vector<int> & verticesToRemoveRef){
        
        compressedSparseMatrix->GetNewColRef().resize(edgesLeftToCover);
        // Temporary, used for checking, will be replaced by vector of all 1's

        std::vector<int> & row_offsets = compressedSparseMatrix->GetOldRowOffRef();
        std::vector<int> & column_indices = compressedSparseMatrix->GetOldColRef();
        std::vector<int> & values = compressedSparseMatrix->GetNewValRef();
        
        std::vector<int> & newRowOffsets = compressedSparseMatrix->GetNewRowOffRef();
        std::vector<int> & newColumnIndices = compressedSparseMatrix->GetNewColRef();
        //std::vector<int> & newValuesRef;

        newValues.resize(edgesLeftToCover);
        SetNewRowOffsets(newRowOffsets);
        int row; 
        
        #pragma omp parallel for default(none) \
                            shared(row_offsets, column_indices, values, \
                            new_degrees, newRowOffsets, newColumnIndices, newValues) \
                            private (row)
        for (row = 0; row < vertexCount; ++row)
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
        std::cout << "removing the vertices in verticesToRemoveRef: " << std::endl;
        for (auto & v : verticesToRemoveRef)
            std::cout << v << " ";
        std::cout << std::endl;
        std::cout << "Along with all their neighbors that are now deg 0 " << std::endl;
        RemoveNewlyDegreeZeroVertices(verticesToRemoveRef, row_offsets, column_indices, newRowOffsets);
        std::cout << "Done" << std::endl;

        // We dont need the values array anymore
        // Reset with a small vector of all 1's
        // Maybe a temporary sol'n
        values.clear();
        values.resize(edgesLeftToCover, 1);
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

void Graph::BuildTheExampleCOO(COO * coordinateFormat){
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

    coordinateFormat->size = coordinateFormat->new_values.size();
    // vlog(e)
    coordinateFormat->sortMyself();
}

void Graph::BuildCOOFromFile(COO * coordinateFormat, std::string filename){
    char sep = ' ';
    std::ifstream       file(filename);
    std::string::size_type sz;   // alias of size_t
    for(auto& row: CSVRange(file, sep))
    {
        //std::cout << "adding (" << std::stoi(row[0],&sz) 
        //<< ", " << std::stoi(row[1],&sz) << ")" << std::endl; 
        coordinateFormat->addEdgeSymmetric(std::stoi(row[0],&sz), 
                                            std::stoi(row[1],&sz), 1);
    }

    coordinateFormat->size = coordinateFormat->new_values.size();
    // vlog(e)
    coordinateFormat->sortMyself();
}

std::vector<int> & Graph::GetNewDegRef(){
    return new_degrees;
}

void Graph::PrintEdgesOfS(){
    std::cout << "E(S) = {";
    int i, u, v;
    for (u = 0; u < vertexCount; ++u){
        for (i = compressedSparseMatrix->old_row_offsets_ref[u]; i < compressedSparseMatrix->old_row_offsets_ref[u+1]; ++i){
            v = compressedSparseMatrix->old_column_indices_ref[i];
            if(!compressedSparseMatrix->new_values[i])
                std::cout << "(" << u << ", " << v << "), ";
        }
    }
    std::cout << "}" << std::endl;
}

bool Graph::GPrimeEdgesGreaterKTimesKPrime(int k, int kPrime){
    int kTimesKPrime = k * kPrime;
    if (edgesLeftToCover/2 > kTimesKPrime)
        return true;
    return false;
}

int Graph::GetRandomVertex(){
    std::cout << "verticesRemaining.size() " << verticesRemaining.size() << std::endl;
    int index = rand() % verticesRemaining.size();
    return verticesRemaining[index];
}
