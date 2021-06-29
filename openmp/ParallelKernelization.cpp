#include "ParallelKernelization.h"
#include<cmath>


ParallelKernelization::ParallelKernelization(Graph & g_arg, int k_arg):g(g_arg), k(k_arg){
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printf(" hello(%d) ", ID);
        printf(" world(%d) \n", ID);
    }
    numberOfProcessors = omp_get_max_threads();
    numberOfElements = g.GetCOO()->row_indices.size();
    blockSize = numberOfElements/numberOfProcessors;
    rowBlockSize = g.GetCSR()->numberOfRows/numberOfProcessors;
    numberOfRows = g.GetCSR()->numberOfRows;
    std::vector<int> degrees(g.GetCSR()->numberOfRows);
    std::vector<int> vertexKeys(g.GetCSR()->numberOfRows);
    std::iota (std::begin(vertexKeys), std::end(vertexKeys), 0); // Fill with 0, 1, ..., 99.
    for (int i = 0; i < numberOfRows; ++i){
        degrees[i] = g.GetCSR()->row_offsets[i+1] - g.GetCSR()->row_offsets[i];
    }

    std::cout << "degrees " << std::endl;
    for (auto & v : degrees)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "vertexKeys " << std::endl;
    for (auto & v : vertexKeys)
        std::cout << v << " ";
    std::cout << std::endl;

    int max = 0;
    for (int i = 0; i < numberOfRows; ++i){
        if (degrees[i] > max)
            max = degrees[i];
    }

    std::vector<int> & A_row_indices = degrees;
    std::vector<int> & A_col_indices = vertexKeys;
    std::vector<int> & A_values = vertexKeys;

    std::cout << "Max : " << max << std::endl;

    B_row_indices.resize(numberOfRows);
    B_column_indices.resize(numberOfRows);
    B_values.resize(numberOfRows);

    C.resize(max+1, 0);
/*
    B_row_indices.resize(numberOfElements);
    B_column_indices.resize(numberOfElements);
    B_values.resize(numberOfElements);



    std::cout << "C" << std::endl;
    for (auto & v : C)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "Before sort" << std::endl;
    for (auto & v : g.GetCOO()->row_indices)
        std::cout << v << " ";
    std::cout << std::endl;

    for (auto & v : g.GetCOO()->column_indices)
        std::cout << v << " ";
    std::cout << std::endl;

    for (auto & v : g.GetCOO()->values)
        std::cout << v << " ";
    std::cout << std::endl;
*/
    // Sort by degree
    CountingSortSerial(   max,
                    degrees,
                    vertexKeys,
                    vertexKeys,
                    B_row_indices,
                    B_column_indices,
                    B_values,
                    C
                );
/*
    std::vector<std::vector<int>> testVectorsRows(numberOfProcessors);
    std::vector<std::vector<int>> testVectorsColumns(numberOfProcessors);
    std::vector<std::vector<int>> testVectorsValues(numberOfProcessors);

    #pragma omp parallel for default(none) shared(testVectorsRows, \
                    testVectorsColumns, testVectorsValues, \
                    A_row_indices, A_col_indices, A_values)
    //shared(boxAxes, cellStartIndex, \
    //reduction(+:tempREn, tempLJEn)
    for (int i = 0; i < numberOfProcessors; ++i)
    {
        int procID = omp_get_thread_num();
        printf(" hello(%d) ", procID);
        printf(" world(%d) \n", procID);

    testVectorsRows[procID].resize(GetEndingIndexInA(procID) - GetStartingIndexInA(procID));
    testVectorsColumns[procID].resize(GetEndingIndexInA(procID) - GetStartingIndexInA(procID));
    testVectorsValues[procID].resize(GetEndingIndexInA(procID) - GetStartingIndexInA(procID));
*/
/*
    printf("thread (%d) : size (%lu)\n", procID, testVectorsRows[procID].size());
    CountingSortParallel(
                    procID,
                    GetStartingIndexInA(procID),
                    GetEndingIndexInA(procID),
                    A_row_indices,
                    A_col_indices,
                    A_values,
                    testVectorsRows[procID],
                    testVectorsColumns[procID],
                    testVectorsValues[procID]);

    }

    printf(" Calling RadixSortWrapper (%d) ", procID);
    ParallelRadixSortWrapper(
                    procID,
                    GetStartingIndexInA(procID),
                    GetEndingIndexInA(procID),
                    A_row_indices,
                    A_col_indices,
                    A_values,
                    testVectorsRows[procID],
                    testVectorsColumns[procID],
                    testVectorsValues[procID]);
    }
    
    for (int i = 0; i < numberOfProcessors; ++i)
    {
        std::cout << "proc " << i << std::endl;
        for (auto & v : testVectorsRows[i])
            std::cout << v << " ";
        std::cout << std::endl;
    }


    std::cout << "After sort" << std::endl;
    for (int i = 0; i < B_row_indices.size(); ++i)
        std::cout << B_row_indices[i] << " ";
    std::cout << std::endl;

    for (int i = 0; i < B_row_indices.size(); ++i)
        std::cout << B_column_indices[i] << " ";
    std::cout << std::endl;
*/

    std::cout << "degrees " << std::endl;
    for (auto & v : B_row_indices)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "vertexKeys " << std::endl;
    for (auto & v : B_column_indices)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "Build VC" << std::endl;
    noSolutionExists = CardinalityOfSetDegreeGreaterK(B_row_indices, B_column_indices);
    printf("%s\n", noSolutionExists ? "b > k, no solution exists" : "b <= k, a solution may exist");
    if (noSolutionExists)
        exit(0);
    PrintS();            
    std::cout << "Removing S from G" << std::endl;
    //SetEdgesOfS(g.GetCSR());
    SetEdgesOfSSym(g.GetCSR());
    //PrintEdgesOfS();
    SetEdgesLeftToCover(g.GetCSR());
    std::cout << g.edgesLeftToCover << " edges left in induced subgraph G'" << std::endl;
    kPrime = k - b;
    std::cout << "Setting k' = k - b = " << kPrime << std::endl;
    noSolutionExists = GPrimeEdgesGreaterKTimesKPrime();
    printf("%s\n", noSolutionExists ? "|G'(E)| > k*k', no solution exists" : "|G'(E)| <= k*k', a solution may exist");          
}

void ParallelKernelization::CountingSortSerial(int max,
                        std::vector<int> & A_row_indices,
                        std::vector<int> & A_column_indices,
                        std::vector<int> & A_values,
                        std::vector<int> & B_row_indices_ref,
                        std::vector<int> & B_column_indices_ref,
                        std::vector<int> & B_values_ref,
                        std::vector<int> & C_ref)
{
    for (int i = 0; i < A_row_indices.size(); ++i){
        ++C_ref[A_row_indices[i]];
    }

    std::cout << "C[i] now contains the number elements equal to i." << std::endl;
    for (int i = 1; i < max+1; ++i){
        C_ref[i] = C_ref[i] + C_ref[i-1];
    }

    std::cout << "C[i] now contains the number of elements less than or equal to i." << std::endl;
    /* C_ref[A_row_indices[i]]]-1 , because the values of C_ref are from [1, n] -> [0,n)*/
    for (int i = B_row_indices.size()-1; i >= 0; --i){
        B_row_indices_ref[C_ref[A_row_indices[i]]-1] = A_row_indices[i];
        B_column_indices_ref[C_ref[A_row_indices[i]]-1] = A_column_indices[i];
        B_values_ref[C_ref[A_row_indices[i]]-1] = A_values[i];
        --C_ref[A_row_indices[i]];
    }
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

int ParallelKernelization::GetStartingIndexInA(int processorID){
    return processorID*blockSize;
}

int ParallelKernelization::GetEndingIndexInA(int processorID){
    if (processorID == numberOfProcessors-1)
        return numberOfElements;
    else
        return (processorID+1)*blockSize;
}

int ParallelKernelization::GetBlockSize(){
    return blockSize;
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

void ParallelKernelization::PrintEdgesOfS(){
    std::cout << "E(S) = {";
    for (auto & e : g.edgesCoveredByKernelization){
        std::cout << "(" << e.first << ", " << e.second << "), ";
    }
    std::cout << "}" << std::endl;
}

void ParallelKernelization::SetEdgesOfS(CSR * csr){
    int v;
    for (auto u : S){
        for (int i = csr->row_offsets[u]; i < csr->row_offsets[u+1]; ++i){
            v = csr->column_indices[i];
            if (u < v){
                g.edgesCoveredByKernelization.insert(std::make_pair(u,v));
            } else {
                g.edgesCoveredByKernelization.insert(std::make_pair(v,u));
            }
        }
    }
}

void ParallelKernelization::SetEdgesOfSSym(CSR * csr){
    int v;
    std::vector<int>::iterator low;

    // We mask all the edges of each vertex in S, 
    // then we mask the in-edges of all vertices in S
    // It is possible there is overlap between these.
    // So we may need atomic operations for in-edges if we parallelize
    // We are just overwriting 1 to 0 though, so it's less sensitive 
    // to race conditions, just have to be concerned about data corruption
    // This is also why we cant simply decrement a degree counter,
    // In case a vertex's in edge is another vertex's out edge
    
    for (auto u : S){
        for (int i = csr->row_offsets[u]; i < csr->row_offsets[u+1]; ++i){
            v = csr->column_indices[i];
            //!!!!!   a must be sorted by cols within rows.       
            low = std::lower_bound( csr->column_indices.begin() + csr->row_offsets[v], 
                                    csr->column_indices.begin() + csr->row_offsets[v+1], 
                                    u);
            csr->values[i] = 0;
            int tmp = low - (csr->column_indices.begin() + csr->row_offsets[v]);
            std::cout << "tmp " << tmp << std::endl;
            csr->values[csr->row_offsets[v] + (low - (csr->column_indices.begin() + csr->row_offsets[v]))] = 0;
        }
    }
}

int ParallelKernelization::GetCardinalityOfSEdges(){
    return g.edgesCoveredByKernelization.size();
}

int ParallelKernelization::GetKPrime(){
    return kPrime;
}


void ParallelKernelization::SetEdgesLeftToCover(CSR * csr){
    int count = 0;
    for (int i = 0; i < numberOfElements; ++i)
        count += csr->values[i];

    g.edgesLeftToCover = count;
}

bool ParallelKernelization::GPrimeEdgesGreaterKTimesKPrime(){
    int kTimesKPrime = k * kPrime;
    if (g.edgesLeftToCover/2 > kTimesKPrime)
        return true;
    return false;
}
