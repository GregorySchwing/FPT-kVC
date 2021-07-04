#include "ParallelKernelization.h"


ParallelKernelization::ParallelKernelization(Graph & g_arg, int k_arg):g(g_arg), k(k_arg)
{
    std::cout << "Entered PK" << std::endl;

    numberOfRows = g.GetCSR()->numberOfRows;

    std::vector<int> & old_degree_ref = g_arg.GetNewDegRef();

    LinearTimeDegreeSort ltds(numberOfRows, old_degree_ref);


    std::cout << "Build VC" << std::endl;
    noSolutionExists = CardinalityOfSetDegreeGreaterK(ltds.GetDegreeRef(), ltds.GetVertexKeyRef());
    printf("%s\n", noSolutionExists ? "b > k, no solution exists" : "b <= k, a solution may exist");
    if (noSolutionExists)
        exit(0);
    PrintS();            
    std::cout << "Removing S from G" << std::endl;


    /*
    vertexTouchedByRemovedEdge.resize(numberOfRows);
    SetEdgesOfSSymParallel();
    SetEdgesLeftToCoverParallel();
    PrintEdgesOfS();
    std::cout << g.edgesLeftToCover << " edges left in induced subgraph G'" << std::endl;
    kPrime = k - b;
    std::cout << "Setting k' = k - b = " << kPrime << std::endl;
    noSolutionExists = GPrimeEdgesGreaterKTimesKPrime();
    if(noSolutionExists)
        std::cout << "|G'(E)| > k*k', no solution exists" << std::endl;
    else{
        std::cout << "|G'(E)| <= k*k', a solution may exist" << std::endl;

        newColumnIndices.resize(g.edgesLeftToCover);
        // Temporary, used for checking, will be replaced by vector of all 1's
        newValues.resize(g.edgesLeftToCover);
        //newColumnIndices.resize(26);
        // Temporary, used for checking, will be replaced by vector of all 1's
        //newValues.resize(26);
        SetNewRowOffsets();

        int row; 
        
        #pragma omp parallel for default(none) \
                            shared( row_offsets, column_indices, values, \
                                    newRowOffsets, newColumnIndices, newValues) \
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
        RemoveDegreeZeroVertices();
        //RemoveSVertices();
    }*/
}


void ParallelKernelization::CountingSortParallel(
                int procID,
                int beginIndex,
                int endIndex,
                std::vector<int> & A_row_indices,
                std::vector<int> & A_column_indices,
                std::vector<int> & A_values,
                std::vector<int> & B_row_indices_ref,
                std::vector<int> & B_column_indices_ref,
                std::vector<int> & B_values_ref){

    //std::cout << "procID : " << procID << " beginIndex " << beginIndex << " endIndex " << endIndex << std::endl;

    int max = 0;
    for (int i = beginIndex; i < endIndex; ++i){
        if (A_row_indices[i] > max)
            max = A_row_indices[i];
    }



    std::vector<int> C_ref(max+1, 0);

    for (int i = beginIndex; i < endIndex; ++i){
        ++C_ref[A_row_indices[i]];
    }

    //std::cout << "C[i] now contains the number elements equal to i." << std::endl;
    for (int i = 1; i < max+1; ++i){
        C_ref[i] = C_ref[i] + C_ref[i-1];
    }

    //std::cout << "C[i] now contains the number of elements less than or equal to i." << std::endl;

    /* C_ref[A_row_indices[i]]]-1 , because the values of C_ref are from [1, n] -> [0,n)*/
    for (int i = endIndex; i > beginIndex; --i){
        B_row_indices_ref[C_ref[A_row_indices[i]]-1] = A_row_indices[i];
        B_column_indices_ref[C_ref[A_row_indices[i]]-1] = A_column_indices[i];
        B_values_ref[C_ref[A_row_indices[i]]-1] = A_values[i];
        --C_ref[A_row_indices[i]];
    }
}

void ParallelKernelization::CountingSortParallelRowwiseValues(
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

void ParallelKernelization::ParallelRadixSortWrapper(int procID,
                int beginIndex,
                int endIndex,
                std::vector<int> & A_row_indices,
                std::vector<int> & A_column_indices,
                std::vector<int> & A_values,
                std::vector<int> & B_row_indices_ref,
                std::vector<int> & B_column_indices_ref,
                std::vector<int> & B_values_ref){

    /* Get longest integer length */
    int maxLength = 0, size = 0;
    for (int i = beginIndex; i < endIndex; ++i){
        if (A_row_indices[i] == 0)
            size = 1;
        else
            size = trunc(log10(A_row_indices[i])) + 1;
        if (size > maxLength)
            maxLength = size;
    }

    std::cout << "MAXL" << maxLength << std::endl;

    int base = 10;
    int digit;
    std::vector<int> C_ref(base+1, 0);

    for (int digit = 0; digit < maxLength; ++digit){
        ParallelRadixSortWorker(procID,
                        beginIndex,
                        endIndex,
                        digit,
                        base,
                        A_row_indices,
                        A_column_indices,
                        A_values,
                        B_row_indices_ref,
                        B_column_indices_ref,
                        B_values_ref,
                        C_ref);
    }
}

void ParallelKernelization::ParallelRadixSortWorker(int procID,
                int beginIndex,
                int endIndex,
                int digit,
                int base,
                std::vector<int> & A_row_indices,
                std::vector<int> & A_column_indices,
                std::vector<int> & A_values,
                std::vector<int> & B_row_indices_ref,
                std::vector<int> & B_column_indices_ref,
                std::vector<int> & B_values_ref,
                std::vector<int> & C_ref){

    C_ref.clear();
    int entry;
    for (int i = beginIndex; i < endIndex; ++i){
        if (digit == 0)
            entry = A_row_indices[i] % base;
        else
            entry = (A_row_indices[i]/(digit*base)) % base;

        std::cout << "entry " << entry << std::endl;
        ++C_ref[entry];
    }

    std::cout << "C[i] now contains the number elements equal to i." << std::endl;
    for (int i = 1; i < base+1; ++i){
        C_ref[i] = C_ref[i] + C_ref[i-1];
    }

    /* C_ref[A_row_indices[i]]]-1 , because the values of C_ref are from [1, n] -> [0,n)*/
    for (int i = endIndex; i > beginIndex; --i){
        B_row_indices_ref[C_ref[A_row_indices[i]]-1] = A_row_indices[i];
        B_column_indices_ref[C_ref[A_row_indices[i]]-1] = A_column_indices[i];
        B_values_ref[C_ref[A_row_indices[i]]-1] = A_values[i];
        --C_ref[A_row_indices[i]];
    }

    for (int i = endIndex; i > beginIndex; --i){
        A_row_indices[i] = B_row_indices_ref[i];
        A_column_indices[i] = B_column_indices_ref[i];
        A_values[i] = B_values_ref[i];
    }

}

bool ParallelKernelization::CardinalityOfSetDegreeGreaterK(std::vector<int> & degrees,
                                                           std::vector<int> & vertexKeys){
    b = GetSetOfVerticesDegreeGreaterK(degrees, vertexKeys);
    if (b > k)
        return true;
    else
        return false;
}

/* Use the Count function of dynamic bitset */
int ParallelKernelization::GetSetOfVerticesDegreeGreaterK(std::vector<int> & degrees,
                                                           std::vector<int> & vertexKeys){    
    S.clear();
    std::vector<int>::iterator up;
    up=std::upper_bound (degrees.begin(), degrees.end(), k); // 
    int cardinalityOfS = degrees.end() - up;
    std::cout << "cardinality of B " << (degrees.end() - up) << '\n'; 
    std::vector<int>::iterator upCopy(up);
 
    while(upCopy != degrees.end()){
        S.push_back(vertexKeys[upCopy - degrees.begin()]);
        upCopy++;
    }
    return cardinalityOfS;
}

std::vector<int> & ParallelKernelization::GetS(){
    return S;
}


void ParallelKernelization::PrintS(){
    std::cout << "S = {";
    for (auto & i : S){
        std::cout << i << " ";
    }
    std::cout << "}" << std::endl;
}

int ParallelKernelization::GetKPrime(){
    return kPrime;
}
/*

void ParallelKernelization::PrintEdgesOfS(){
    std::cout << "E(S) = {";
    int i, u, v;
    for (u = 0; u < numberOfRows; ++u){
        for (i = row_offsets[u]; i < row_offsets[u+1]; ++i){
            v = column_indices[i];
            if(!values[i])
                std::cout << "(" << u << ", " << v << "), ";
        }
    }
    std::cout << "}" << std::endl;
}

int ParallelKernelization::GetRandomVertex(){
    int index = rand() % verticesRemaining.size();
    return verticesRemaining[index];
}

void ParallelKernelization::SetEdgesOfSSym(){
    int v;
    std::vector<int>::iterator low;
    
    for (auto u : S){
        for (int i = row_offsets[u]; i < row_offsets[u+1]; ++i){
            v = column_indices[i];
            //!!!!!   a must be sorted by cols within rows.       
            low = std::lower_bound( column_indices.begin() + row_offsets[v], 
                                    column_indices.begin() + row_offsets[v+1], 
                                    u);
            values[i] = 0;
            values[row_offsets[v] + (low - (column_indices.begin() + row_offsets[v]))] = 0;
        }
    }
}

void ParallelKernelization::SetEdgesOfSSymParallel(){
    int v, intraRowOffset;
    std::vector<int>::iterator low;
    // Set out-edges
    #pragma omp parallel for default(none) shared(row_offsets, \
    column_indices, values) private (v)
    for (auto u : S)
    {
        for (int i = row_offsets[u]; i < row_offsets[u+1]; ++i){
            v = column_indices[i];
            values[i] = 0;
        }
    }

    // Set in-edges
    #pragma omp parallel for default(none) shared(row_offsets, \
    column_indices, values) private (low, v, intraRowOffset)
    for (auto u : S)
    {
        for (int i = row_offsets[u]; i < row_offsets[u+1]; ++i){
            v = column_indices[i];
            // Break this into 2 independent for loops 
            //!!!!!   a must be sorted by cols within rows.       
            low = std::lower_bound( column_indices.begin() + row_offsets[v], 
                                    column_indices.begin() + row_offsets[v+1], 
                                    u);
            intraRowOffset = low - (column_indices.begin() + row_offsets[v]);
            // Set in-edge
            values[row_offsets[v] + intraRowOffset] = 0;
        }
    }
}

void ParallelKernelization::RemoveSVertices(){
    for (auto u : S)
    {
        g.removeVertex(u, verticesRemaining);
    }
}

void ParallelKernelization::RemoveDegreeZeroVertices(){
    for (int i = 0; i < numberOfRows; ++i){
        if(newRowOffsets[i+1] - newRowOffsets[i] == 0)
            g.removeVertex(i, verticesRemaining);
    }
}


void ParallelKernelization::SetEdgesLeftToCover(){
    int count = 0;
    for (int i = 0; i < numberOfElements; ++i)
        count += values[i];

    g.edgesLeftToCover = count;
}

void ParallelKernelization::SetEdgesLeftToCoverParallel(){
    int count = 0, i = 0, j = 0;
    std::vector<int> & newDegs = newDegrees;
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

void ParallelKernelization::SetNewRowOffsets(){
    int i = 0;
    std::vector<int> & newDegs = newDegrees;
    std::vector<int> & newRowOffs = newRowOffsets;
    newRowOffs.resize(numberOfRows+1);
    for (i = 1; i <= numberOfRows; ++i)
    {
        newRowOffs[i] = newDegs[i-1] + newRowOffs[i-1];
    }
}

bool ParallelKernelization::GPrimeEdgesGreaterKTimesKPrime(){
    int kTimesKPrime = k * kPrime;
    if (g.edgesLeftToCover/2 > kTimesKPrime)
        return true;
    return false;
}

std::vector<int> & ParallelKernelization::GetRowOffRef(){
    return row_offsets;
}
std::vector<int> & ParallelKernelization::GetColRef(){
    return column_indices;
}
std::vector<int> & ParallelKernelization::GetValRef(){
    return values;
}
std::vector<int> & ParallelKernelization::GetVerticesRemainingRef(){
    return verticesRemaining;
}
*/
int ParallelKernelization::GetNumberOfRows(){
    return numberOfRows;
}

