#include "SequentialKernelization.h"

SequentialKernelization::SequentialKernelization(Graph & g_arg, int k_arg):g(g_arg), k(k_arg){
    std::cout << "Build VC" << std::endl;
    noSolutionExists = CardinalityOfSetDegreeGreaterK(g.GetDegreeController());
    printf("%s\n", noSolutionExists ? "b > k, no solution exists" : "b <= k, a solution may exist");
    if (noSolutionExists)
        exit(0);
    PrintS();            
    std::cout << "Removing S from G" << std::endl;
    //RemoveSFromG();
    SetEdgesOfS(g.GetCSR());
    PrintEdgesOfS();
    SetEdgesLeftToCover();
    std::cout << g.edgesLeftToCover << " edges left in induced subgraph G'" << std::endl;
    kPrime = k - b;
    std::cout << "Setting k' = k - b = " << kPrime << std::endl;
    noSolutionExists = GPrimeEdgesGreaterKTimesKPrime();
    printf("%s\n", noSolutionExists ? "|G'(E)| > k*k', no solution exists" : "|G'(E)| <= k*k', a solution may exist");
                
}

bool SequentialKernelization::CardinalityOfSetDegreeGreaterK(DegreeController * degCont){
    S.clear();
    b = GetSetOfVerticesDegreeGreaterK(k, S, degCont);
    if (b > k)
        return true;
    else
        return false;
}

/* Use the Count function of dynamic bitset */
int SequentialKernelization::GetSetOfVerticesDegreeGreaterK(int k, std::vector<int> & S, DegreeController * degCont){
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

std::vector<int> & SequentialKernelization::GetS(){
    return S;
}


void SequentialKernelization::PrintS(){
    std::cout << "S = {";
    for (auto & i : S){
        std::cout << i << " ";
    }
    std::cout << "}" << std::endl;
}

void SequentialKernelization::PrintEdgesOfS(){
    std::cout << "E(S) = {";
    for (auto & e : g.edgesCoveredByKernelization){
        std::cout << "(" << e.first << ", " << e.second << "), ";
    }
    std::cout << "}" << std::endl;
}

void SequentialKernelization::SetEdgesOfS(CSR * csr){
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

int SequentialKernelization::GetCardinalityOfSEdges(){
    return g.edgesCoveredByKernelization.size();
}

int SequentialKernelization::GetKPrime(){
    return kPrime;
}


void SequentialKernelization::SetEdgesLeftToCover(){
    g.edgesLeftToCover -= GetCardinalityOfSEdges();
}

bool SequentialKernelization::GPrimeEdgesGreaterKTimesKPrime(){
    int kTimesKPrime = k * kPrime;
    if (g.edgesLeftToCover > kTimesKPrime)
        return true;
    return false;
}
