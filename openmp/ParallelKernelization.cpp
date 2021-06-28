#include "ParallelKernelization.h"

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

    int max = 0;
    for (int i = 0; i < numberOfElements; ++i){
        if (g.GetCOO()->row_indices[i] > max)
            max = g.GetCOO()->row_indices[i];
    }

    std::vector<int> & A_row_indices = g.GetCOO()->row_indices;
    std::vector<int> & A_col_indices = g.GetCOO()->column_indices;
    std::vector<int> & A_values = g.GetCOO()->values;

    std::cout << "Max : " << max << std::endl;

    B_row_indices.resize(numberOfElements);
    B_column_indices.resize(numberOfElements);
    B_values.resize(numberOfElements);

    C.resize(max+1, 0);

/*

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

    CountingSortSerial(   max,
                    A_row_indices,
                    A_col_indices,
                    A_values,
                    B_row_indices,
                    B_column_indices,
                    B_values,
                    C
                );

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

    for (int i = 0; i < numberOfProcessors; ++i)
    {
        std::cout << "proc " << i << std::endl;
        for (auto & v : testVectorsRows[i])
            std::cout << v << " ";
        std::cout << std::endl;
    }
/*
    std::cout << "After sort" << std::endl;
    for (int i = 0; i < B_row_indices.size(); ++i)
        std::cout << B_row_indices[i] << " ";
    std::cout << std::endl;

    for (int i = 0; i < B_row_indices.size(); ++i)
        std::cout << B_column_indices[i] << " ";
    std::cout << std::endl;

    for (int i = 0; i < B_row_indices.size(); ++i)
        std::cout << B_row_indices[i] << " ";
    std::cout << std::endl;
*/
    /*
    std::cout << "Build VC" << std::endl;
    noSolutionExists = CardinalityOfSetDegreeGreaterK(g.GetDegreeController());
    printf("%s\n", noSolutionExists ? "b > k, no solution exists" : "b <= k, a solution may exist");
    if (noSolutionExists)
        exit(0);
    PrintS();            
    std::cout << "Removing S from G" << std::endl;
    //SetEdgesOfS(g.GetCSR());
    SetEdgesOfSSym(g.GetCSR());
    PrintEdgesOfS();
    SetEdgesLeftToCover();
    std::cout << g.edgesLeftToCover << " edges left in induced subgraph G'" << std::endl;
    kPrime = k - b;
    std::cout << "Setting k' = k - b = " << kPrime << std::endl;
    noSolutionExists = GPrimeEdgesGreaterKTimesKPrime();
    printf("%s\n", noSolutionExists ? "|G'(E)| > k*k', no solution exists" : "|G'(E)| <= k*k', a solution may exist");
    */            
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
    for (int i = 0; i < numberOfElements; ++i){
        ++C_ref[A_row_indices[i]];
    }

    //std::cout << "C[i] now contains the number elements equal to i." << std::endl;
    for (int i = 1; i < max+1; ++i){
        C_ref[i] = C_ref[i] + C_ref[i-1];
    }

    //std::cout << "C[i] now contains the number of elements less than or equal to i." << std::endl;

    /* C_ref[A_row_indices[i]]]-1 , because the values of C_ref are from [1, n] -> [0,n)*/
    for (int i = B_row_indices.size(); i > 0; --i){
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

void ParallelKernelization::RadixSort(){

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

bool ParallelKernelization::CardinalityOfSetDegreeGreaterK(DegreeController * degCont){
    S.clear();
    b = GetSetOfVerticesDegreeGreaterK(k, S, degCont);
    if (b > k)
        return true;
    else
        return false;
}

/* Use the Count function of dynamic bitset */
int ParallelKernelization::GetSetOfVerticesDegreeGreaterK(int k, std::vector<int> & S, DegreeController * degCont){
    std::vector< std::vector<int> > & tempDegCont = degCont->GetTempDegCont();
    std::vector< std::vector<int> >::reverse_iterator it = tempDegCont.rbegin();
    int cardinalityOfS = 0;
    int iteration = 0;
    // This scans the degree controller from N-1 to k + 1.
    // We construct S, which is the set of vertices with deg > k
    // Hence we iterate over all vertices with degree(N-1) to k + 1
    // We early terminate is |S| > k
    while(it != (tempDegCont.rend() - k - 1)  && cardinalityOfS <= k){
        std::cout << "Iteration " << iteration << " (vertices w degree " << tempDegCont.size() - iteration - 1 << " ) : ";
        //appending elements of vector of vertices of deg(x) to vector S
        // while deg(x) > k and cardinalityOfS <= k
        for (auto & e : *it)
            std::cout << e << " ";
        std::cout <<std::endl;
        S.insert(S.end(), it->begin(), it->end());
        cardinalityOfS+=it->size();
        iteration++;
        it++;
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
    for (auto u : S){
        for (int i = csr->row_offsets[u]; i < csr->row_offsets[u+1]; ++i){
            v = csr->column_indices[i];
            g.edgesCoveredByKernelization.insert(std::make_pair(u,v));
        }
    }
}

int ParallelKernelization::GetCardinalityOfSEdges(){
    return g.edgesCoveredByKernelization.size();
}

int ParallelKernelization::GetKPrime(){
    return kPrime;
}


void ParallelKernelization::SetEdgesLeftToCover(){
    g.edgesLeftToCover -= GetCardinalityOfSEdges();
}

bool ParallelKernelization::GPrimeEdgesGreaterKTimesKPrime(){
    int kTimesKPrime = k * kPrime;
    if (g.edgesLeftToCover > kTimesKPrime)
        return true;
    return false;
}
