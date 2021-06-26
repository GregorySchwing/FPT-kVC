#include "ParallelKernelization.h"

ParallelKernelization::ParallelKernelization(Graph & g_arg, int k_arg):g(g_arg), k(k_arg){
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printf(" hello(%d) ", ID);
        printf(" world(%d) \n", ID);
    }
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
