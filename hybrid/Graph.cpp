#include "Graph.h"

/* Create first Graph */
Graph::Graph(CSR & csr_arg):
    csr(csr_arg),
    old_degrees_ref(&old_degrees),
    vertexCount(csr_arg.vertexCount)
{
    std::cout << "First G" << std::endl;
    ProcessGraph(csr.numberOfRows);
}

Graph::Graph(const Graph & other): csr(other.csr),
    vertexCount(other.vertexCount){
    hasntBeenRemoved.reserve(other.vertexCount);
    verticesRemaining.reserve(other.vertexCount);
    new_degrees.reserve(other.vertexCount);
//    std::cout << "Copied" << std::endl;
}


int Graph::GetSize(){
    return GetCSR().GetSize();
}

int Graph::GetNumberOfRows(){
    return GetCSR().GetNumberOfRows();
}

thrust::host_vector<int> & Graph::GetVerticesThisGraphIncludedInTheCover(){
    return verticesToIncludeInCover;
}

// This approach will continually append vertices if we keep processing
// immediately in the case of pendant edges
void Graph::SetVerticesToIncludeInCover(thrust::host_vector<int> & verticesRef){
    verticesToIncludeInCover.insert(std::end(verticesToIncludeInCover), 
        std::begin(verticesRef), std::end(verticesRef));
}

void Graph::InitG(Graph & g_parent, thrust::host_vector<int> & S){
    PopulatePreallocatedMemoryFirstGraph(g_parent);
    SetEdgesOfSSymParallel(S, *(GetCSR().GetOldRowOffRef()), *(GetCSR().GetOldColRef())); 
    SetEdgesLeftToCoverParallel(*(GetCSR().GetOldRowOffRef()));
    RemoveNewlyDegreeZeroVertices(S, *(GetCSR().GetOldRowOffRef()), *(GetCSR().GetOldColRef()));
    SetVerticesToIncludeInCover(S);
}

/* Constructor to make induced subgraph G' for each branch */
void Graph::InitGPrime(Graph & g_parent, 
                        thrust::host_vector<int> & S)
{        
    if (verticesRemaining.size() != 0)
        std::cout << "error" << std::endl;
    std::cout << "Entered constructor of G induced" << std::endl;
    std::cout << "S :" << std::endl;
    if (!S.empty())
        for (auto v : S){
            std::cout << v << " ";
            std::cout << std::endl;
        }
    // Sets the old references of the new csr 
    // to point to the new references of the argument
    // Pendant edges are processed immediately without spawning children
    // Hence we want to skip tree building, allocation, and inducing 
    // And just remove more edges from the graph
    SetMyOldsToParentsNews(g_parent);
    SetVerticesRemainingAndVerticesRemoved(g_parent);
    PopulatePreallocatedMemory(g_parent);
    InduceSubgraph();
    // Just copy old degrees to new degrees
    if(S.size() != 0){
        SetEdgesOfSSymParallel(S, GetCSR().GetNewRowOffRef(), GetCSR().GetNewColRef()); 
        SetEdgesLeftToCoverParallel(GetCSR().GetNewRowOffRef());
        RemoveNewlyDegreeZeroVertices(S, GetCSR().GetNewRowOffRef(), GetCSR().GetNewColRef());
        SetVerticesToIncludeInCover(S);
    } else {
        new_degrees = GetOldDegRef();
    }
    // This line is throwing an error in valgrind
    // Conditional jump or move depends on uninitialised value(s)
    //std::cout << edgesLeftToCover/2 << " edges left in induced subgraph G'" << std::endl;
}

void Graph::ProcessImmediately(thrust::host_vector<int> & S){
    for(auto & v : new_degrees)
        v = 0;
    SetEdgesOfSSymParallel(S, GetCSR().GetNewRowOffRef(), GetCSR().GetNewColRef()); 
    SetEdgesLeftToCoverParallel(GetCSR().GetNewRowOffRef());
    RemoveNewlyDegreeZeroVertices(S, GetCSR().GetNewRowOffRef(), GetCSR().GetNewColRef());
    //for(auto & v : GetCSR().GetNewRowOffRef())
    //    v = 0;
    //CalculateNewRowOffsets(GetNewDegRef());
    SetVerticesToIncludeInCover(S);
}

void Graph::SetMyOldsToParentsNews(Graph & g_parent){
    this->SetParent(g_parent);
    this->SetOldDegRef(g_parent.GetNewDegRef());
    this->edgesLeftToCover = g_parent.GetEdgesLeftToCover();
    this->GetCSR().SetOldRowOffRef(g_parent.GetCSR().GetNewRowOffRef());
    this->GetCSR().SetOldColRef(g_parent.GetCSR().GetNewColRef());
    this->GetCSR().SetOldValRef(g_parent.GetCSR().GetNewValRef());
}

void Graph::SetVerticesRemainingAndVerticesRemoved(Graph & g_parent){
    this->SetRemainingVertices(g_parent.GetRemainingVerticesRef());
    this->SetHasntBeenRemoved(g_parent.GetHasntBeenRemovedRef());
}

void Graph::PopulatePreallocatedMemory(Graph & g_parent){
    for (int i = 0; i < g_parent.GetVertexCount(); ++i)
        new_degrees.push_back(0);

    this->GetCSR().PopulateNewRefs(g_parent.GetEdgesLeftToCover());
    // In this case, the new offsets needs to calculated
    // In Gprime, old == new
    // Here the old degrees need be recalculated before inducing gNprime
    CalculateNewRowOffsets(GetOldDegRef());
}

void Graph::PopulatePreallocatedMemoryFirstGraph(Graph & g_parent){
    for (int i = 0; i < g_parent.GetVertexCount(); ++i)
        new_degrees.push_back(0);

    this->GetCSR().PopulateNewVals(g_parent.GetEdgesLeftToCover());
    // In this case, the new offsets needs to calculated
    // In Gprime, old == new
    // Here the old degrees need be recalculated before inducing gNprime
    //CalculateNewRowOffsets();
}

void Graph::SetOldDegRef(thrust::host_vector<int> & old_deg_ref_arg){
    old_degrees_ref = &old_deg_ref_arg;
}


void Graph::ProcessGraph(int vertexCount){
    // Coming from the CSR we constructed in the process of building the graph
    thrust::host_vector<int> & old_row_offsets = *(GetCSR().GetOldRowOffRef());
    edgesLeftToCover = GetCSR().GetOldColRef()->size();
    // iterable set of vertices with non-zero degrees
    verticesRemaining.resize(vertexCount);
    std::iota (std::begin(verticesRemaining), std::end(verticesRemaining), 0); // Fill with 0, 1, ..., 99.
    // USed as a constant time lookup for whether a vertex has been removed
    hasntBeenRemoved.resize(vertexCount, 1);
    old_degrees.resize(vertexCount);
    for (int i = 0; i < vertexCount; ++i){
        old_degrees[i] = old_row_offsets[i+1] - old_row_offsets[i];
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
    coordinateFormat->SetNumberOfRows(max+1);
    this->vertexCount = max+1;
    coordinateFormat->vertexCount = max+1;

}

thrust::host_vector<int> Graph::GetRemainingVertices(){
    return verticesRemaining;
}

thrust::host_vector<int> & Graph::GetRemainingVerticesRef(){
    return verticesRemaining;
}

thrust::host_vector<int> Graph::GetHasntBeenRemoved(){
    return hasntBeenRemoved;
}

thrust::host_vector<int> & Graph::GetHasntBeenRemovedRef(){
    return hasntBeenRemoved;
}

void Graph::SetRemainingVertices(thrust::host_vector<int> & verticesRemaining_arg){
    verticesRemaining = verticesRemaining_arg;
}

void Graph::SetHasntBeenRemoved(thrust::host_vector<int> &  hasntBeenRemoved_arg){
    hasntBeenRemoved = hasntBeenRemoved_arg;
}



int Graph::GetEdgesLeftToCover(){
    return edgesLeftToCover;
}


int Graph::GetDegree(int v){
    return (*(csr.old_row_offsets_ref))[v+1] - (*(csr.old_row_offsets_ref))[v];
}

int Graph::GetVertexCount(){
    return vertexCount;
}

std::vector< thrust::host_vector<int> > & Graph::GetChildrenVertices(){
    return childrenVertices;
}


/* The edges */
CSR & Graph::GetCSR(){
    return csr;
}

void Graph::removeVertex(int vertexToRemove){
        
    thrust::host_vector<int>::iterator low;
    low = std::lower_bound( std::begin(verticesRemaining), 
                            std::end(verticesRemaining), 
                            vertexToRemove);

    if(!hasntBeenRemoved[vertexToRemove]){
#ifndef NDEBUG
        // Can occur only if the second child became degree zero after removing the first 
        if (verticesRemaining[low - std::begin(verticesRemaining)] != vertexToRemove || 
            low == std::end(verticesRemaining))
            return;
        else
            std::cout << "Error! Disagreement between verticesRemaining and hasntBeenRemoved!"
                << std::endl;
#else
        return;
#endif

    } else 
        hasntBeenRemoved[vertexToRemove] = 0;


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

void Graph::SetEdgesOfSSymParallel(thrust::host_vector<int> & S,
                                    thrust::host_vector<int> & row_offsets_ref,
                                    thrust::host_vector<int> & column_indices_ref){
    int v, intraRowOffset;
    thrust::host_vector<int>::iterator low;
    int test;
    thrust::host_vector<int> & values = GetCSR().GetNewValRef();
    for (auto u : S)
    {
        if (u < 0 || u > vertexCount){
            std::cout << "error" << std::endl;
            test = -1;
        }
    }

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

// Sets the new degrees without the edges and the edges left to cover
void Graph::SetEdgesLeftToCoverParallel(thrust::host_vector<int> & row_offsets){
    int count = 0, i = 0, j = 0;
    thrust::host_vector<int> & newDegs = new_degrees;
    thrust::host_vector<int> & values = GetCSR().GetNewValRef();
    //thrust::host_vector<int> & row_offsets = 
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
void Graph::CalculateNewRowOffsets(thrust::host_vector<int> & old_degrees){    
    // Cuda load
    // Parent's new degree ref
    //thrust::host_vector<int> & old_degrees = GetOldDegRef();
    // or recalculate degrees if we processed immediately
    thrust::host_vector<int> & new_row_offs = this->GetCSR().GetNewRowOffRef();
    int i = 0;
    new_row_offs.push_back(0);
    for (i = 1; i <= vertexCount; ++i)
    {
        new_row_offs.push_back(old_degrees[i-1] + new_row_offs[i-1]);
    }
}


void Graph::CountingSortParallelRowwiseValues(
                int rowID,
                int beginIndex,
                int endIndex,
                thrust::host_vector<int> & A_row_offsets,
                thrust::host_vector<int> & A_column_indices,
                thrust::host_vector<int> & A_values,
                thrust::host_vector<int> & B_row_indices_ref,
                thrust::host_vector<int> & B_column_indices_ref){
                //,
                //thrust::host_vector<int> & B_values_ref){

    //std::cout << "procID : " << procID << " beginIndex " << beginIndex << " endIndex " << endIndex << std::endl;

    int max = 1;

    thrust::host_vector<int> C_ref(max+1, 0);

    for (int i = beginIndex; i < endIndex; ++i){
        ++C_ref[A_values[i]];
    }

    // This is  [old degree - new degree , new degree]
    for (int i = 1; i < max+1; ++i){
        C_ref[i] = C_ref[i] + C_ref[i-1];
    }
    /* C_ref[A_row_indices[i]]]-1 , because the values of C_ref are from [1, n] -> [0,n) */
    for (int i = endIndex-1; i >= beginIndex; --i){
        if (A_values[i]){
            B_column_indices_ref[B_row_indices_ref[rowID] - C_ref[0] + C_ref[1]-1] = A_column_indices[i];
            //B_values_ref[B_row_indices_ref[rowID] - C_ref[0] + C_ref[1]-1] = A_values[i];
            --C_ref[A_values[i]];
        }
    }
}

// Highly unoptimized, but should work for now
void Graph::RemoveNewlyDegreeZeroVertices(thrust::host_vector<int> & verticesToRemove,
                                            thrust::host_vector<int> & oldRowOffsets,
                                            thrust::host_vector<int> & oldColumnIndices){
    int i = 0, j;
    for (auto & v :verticesToRemove){
        removeVertex(v);
        for (i = oldRowOffsets[v]; i < oldRowOffsets[v+1]; ++i){
            j = oldColumnIndices[i];
            if(new_degrees[j] == 0)
                if (hasntBeenRemoved[j]){
        		std::cout << "removing newly degree zero vertex" << j << std::endl;
	    		removeVertex(j);
                }
        }
    }
}

void Graph::InduceSubgraph(){
       
        // Here in cuda can be repaced by load to shared
        thrust::host_vector<int> & row_offsets = *(csr.GetOldRowOffRef());
        thrust::host_vector<int> & column_indices = *(csr.GetOldColRef());
        // Eventually this line can be commented out
        // and we no longer need to write in parallel in CSPRV
        thrust::host_vector<int> & values = *(GetCSR().GetOldValRef());
        
        thrust::host_vector<int> & newRowOffsets = csr.GetNewRowOffRef();
        thrust::host_vector<int> & newColumnIndices = csr.GetNewColRef();
        // Eventually this line can be commented out
        // and we no longer need to write in parallel in CSPRV

        //testVals.resize(GetEdgesLeftToCover(), 0);

        int row; 
        
        #pragma omp parallel for default(none) \
                            shared(row_offsets, column_indices, values, \
                            new_degrees, newRowOffsets, newColumnIndices, testVals) \
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
                                            newColumnIndices);
                                            //,
                                            //testVals);
        }
}

thrust::host_vector<int> & Graph::GetNewDegRef(){
    return new_degrees;
}

thrust::host_vector<int> * Graph::GetOldDegPointer(){
    return old_degrees_ref;
}

thrust::host_vector<int> & Graph::GetOldDegRef(){
    return *old_degrees_ref;
}

Graph & Graph::GetParent(){
    return *parent;
}


void Graph::PrintEdgesOfS(){
    std::cout << "E(S) = {";
    int i, u, v;
    for (u = 0; u < vertexCount; ++u){
        for (i = (*(csr.old_row_offsets_ref))[u]; i < (*(csr.old_row_offsets_ref))[u+1]; ++i){
            v = (*(csr.old_column_indices_ref))[i];
            if(!csr.new_values[i])
                std::cout << "(" << u << ", " << v << "), ";
        }
    }
    std::cout << "}" << std::endl;
}

void Graph::PrintEdgesRemaining(){
    std::cout << "E(G') = {";
    int i, u, v;
    for (u = 0; u < vertexCount; ++u){
        for (i = (*(csr.old_row_offsets_ref))[u]; i < (*(csr.old_row_offsets_ref))[u+1]; ++i){
            v = (*(csr.old_column_indices_ref))[i];
            if(csr.new_values[i])
                std::cout << "(" << u << ", " << v << "), ";
        }
    }
    std::cout << "}" << std::endl;
}

bool Graph::GPrimeEdgesGreaterKTimesKPrime(int k, int kPrime){
    int kTimesKPrime = k * kPrime;
    std::cout << "Directed edges remaining : " << edgesLeftToCover << std::endl;
    std::cout << "Undirected edges remaining : " << edgesLeftToCover/2 << std::endl;
    if (edgesLeftToCover/2 > kTimesKPrime)
        return true;
    return false;
}

void Graph::SetParent(Graph & g_parent){
    parent = &g_parent;
}

