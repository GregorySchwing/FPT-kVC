#include "SequentialKernelization.h"

SequentialKernelization::SequentialKernelization(Graph & g_arg, int k_arg):g(g_arg), k(k_arg){
    std::cout << "Build VC" << std::endl;
    printf("%s\n",CardinalityOfSetDegreeGreaterK(g.GetDegreeController()) ? "b > k, no solution exists" : "b <= k, a solution may exist");
    PrintS();            
    std::cout << "Removing S from G" << std::endl;
    RemoveSFromG();
    std::cout << g.edgesLeftToCover/2 << " edges left ";
    std::cout << "Setting k' = k - b = " << k - b << std::endl;
    printf("%s\n",GPrimeEdgesGraterKTimesKPrime() ? "G' > k*k', no solution exists" : "G' <= k*k', a solution may exist");
                
}

bool SequentialKernelization::CardinalityOfSetDegreeGreaterK(DegreeController * degCont){
    S.clear();
    b = GetSetOfVerticesDegreeGreaterK(k, S, degCont);
    if (b > k)
        return true;
    else
        return false;
}

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

void SequentialKernelization::PrintS(){
    std::cout << "{";
    for (auto & i : S){
        std::cout << i << " ";
    }
    std::cout << "}" << std::endl;
}
void SequentialKernelization::RemoveSFromG(){
    for (auto v : S)
        g.edgesLeftToCover -= g.GetDegree(v);
}
bool SequentialKernelization::GPrimeEdgesGraterKTimesKPrime(){
    int kTimesKPrime = k * kPrime;
    if (g.edgesLeftToCover/2 > kTimesKPrime)
        return true;
    return false;
}
