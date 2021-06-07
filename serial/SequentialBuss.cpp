#include "SequentialBuss.h"

/* Vertex Cover Using Buss’ algorithm */
SequentialBuss::SequentialBuss(Graph & g_arg, int k_arg, int k_prime_arg):g(g_arg), k(k_arg), k_prime(k_prime_arg){

    /* G(V) - Verts with degree > k = GPrimeVertices <= 2k^2 */
    SetGPrimeVertices();
    /* Hence, GPrimeVertices^k <= (2*k^2)^k */
    
    NumberOfVerticesToThePowerOfKPrime = myPow(verticesOfGPrime.size(), k_prime);
    results = new int[NumberOfVerticesToThePowerOfKPrime];

    /* Hence, the amount of space to store the NumberOfVertices^k combinations
        each of size k, is (NumberOfVertices^k)*k */
    combinations = new int[NumberOfVerticesToThePowerOfKPrime*k_prime];
    PopulateCombinations(combinations, verticesOfGPrime, verticesOfGPrime.size(), k_prime);

    /* The sets of edges covered by each of the NumberOfVertices^k combinations */
    vectorOfSetsOfEdgesCoveredByBuss.resize(NumberOfVerticesToThePowerOfKPrime*k_prime);
    GenerateEdgeSets();
    UnionKernelEdgesAndBFSEdges();
}
SequentialBuss::~SequentialBuss(){
    delete[] results;
    delete[] combinations;
}

void SequentialBuss::SetGPrimeVertices(){
    std::vector< std::vector<int> > & tempDegCont = (g.GetDegreeController())->GetTempDegCont();
    std::vector< std::vector<int> >::const_iterator it = tempDegCont.cbegin();
    while(it != (tempDegCont.cbegin() + k_prime + 2)){
        for (auto & e : *it){
            std::cout << e << " ";
            verticesOfGPrime.push_back(e);
        }
        it++;
    }
}
void SequentialBuss::GenerateEdgeSets(){
    std::cout << "|G'(V)|" << verticesOfGPrime.size() << "k_prime " << k_prime << "|G'(V)|^k_prime " << NumberOfVerticesToThePowerOfKPrime << std::endl;
    /* Iterate through all k_prime-combinations of vertices */
    int u,v;
    for (int x = 0; x < NumberOfVerticesToThePowerOfKPrime; ++x){
        for (int z = 0; z < k_prime; ++z){
            u = combinations[x*k_prime + z];
            for (int i = g.GetCSR()->row_offsets[u]; i < g.GetCSR()->row_offsets[u+1]; ++i){
                v = g.GetCSR()->column_indices[i];
                if (u < v){
                    vectorOfSetsOfEdgesCoveredByBuss[x].insert(std::make_pair(u,v));
                } else {
                    vectorOfSetsOfEdgesCoveredByBuss[x].insert(std::make_pair(v,u));
                }
            }
        }
    }
}

void SequentialBuss::UnionKernelEdgesAndBFSEdges(){
    int totalEdgeCount = g.GetCSR()->column_indices.size();
    for (int x = 0; x < NumberOfVerticesToThePowerOfKPrime; ++x){
        vectorOfSetsOfEdgesCoveredByBuss[x].insert(g.edgesCoveredByKernelization.begin(), g.edgesCoveredByKernelization.end());
        if(vectorOfSetsOfEdgesCoveredByBuss[x].size() - totalEdgeCount == 0)
            results[x] = 1;
        else
            results[x] = 0;
    }
}

void SequentialBuss::PrintVCSets(){
    bool anyAnswerExists = false;
    for (int x = 0; x < NumberOfVerticesToThePowerOfKPrime; ++x){
        if(results[x] != 0){
            anyAnswerExists = true;
            for (int z = 0; z < k_prime; ++z){
                std::cout << " " << combinations[x*k_prime + z];
            }
            std::cout << std::endl;
        }
    }
    if (!anyAnswerExists){
        std::cout << "No k-VC Found" << std::endl;
    } 
}


/* http://rosettacode.org/wiki/Combinations#C.2B.2B */
void SequentialBuss::PopulateCombinations(int * combinations_arg, std::vector<int> & gPrimeVertices, int N, int k_prime){
    std::string bitmask(k_prime, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's
    int rowI = 0;
    int colI = 0;
    // print integers and permute bitmask
    std::cout << std::endl;
    do {
        for (int i = 0; i < N; ++i) // [0..N-1] integers
        {
            if (bitmask[i]){
                /* Row length is K */
                combinations_arg[rowI*k_prime + colI] = gPrimeVertices[i];
                std::cout << " " << gPrimeVertices[i];
                ++colI;
            }
        }
        std::cout << std::endl;
        ++rowI;
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
}

/* https://stackoverflow.com/questions/1505675/power-of-an-integer-in-c */
int SequentialBuss::myPow(int x, unsigned int p)
{
  if (p == 0) return 1;
  if (p == 1) return x;
  
  int tmp = myPow(x, p/2);
  if (p%2 == 0) return tmp * tmp;
    else return x * tmp * tmp;
}