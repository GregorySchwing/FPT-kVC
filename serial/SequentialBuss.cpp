#include "SequentialBuss.h"

/* Vertex Cover Using Buss’ algorithm */
SequentialBuss::SequentialBuss(Graph & g_arg, int k_arg):g(g_arg), k(k_arg){

    /* G(V) - Verts with degree > k = GPrimeVertices <= 2k^2 */
    SetGPrimeVertices();
    /* Hence, GPrimeVertices^k <= (2*k^2)^k */
    
    NumberOfVerticesToThePowerOfK = myPow(verticesOfGPrime.size(), k);
    results = new int[NumberOfVerticesToThePowerOfK];

    /* Hence, the amount of space to store the NumberOfVertices^k combinations
        each of size k, is (NumberOfVertices^k)*k */
    combinations = new int[NumberOfVerticesToThePowerOfK*k];
    PopulateCombinations(combinations, verticesOfGPrime.size(), k);

    /* The sets of edges covered by each of the NumberOfVertices^k combinations */
    vectorOfSetsOfEdgesCoveredByBuss.resize(NumberOfVerticesToThePowerOfK*k);
    GenerateEdgeSets();
    UnionKernelEdgesAndBFSEdges();
}
SequentialBuss::~SequentialBuss(){
    delete[] results;
    delete[] combinations;
}

void SetGPrimeVertices(){
    std::vector< std::vector<int> > & tempDegCont = (g.GetDegreeController())->GetTempDegCont();
    std::vector< std::vector<int> >::const_iterator it = tempDegCont.cbegin();
    while(it != (tempDegCont.cbegin() + k)){
        for (auto & e : *it){
            std::cout << e << " ";
            verticesOfGPrime.push_back(e);
        }
        it++;
    }
}
void SequentialBuss::GenerateEdgeSets(){
    std::cout << "|G'(V)|" << verticesOfGPrime.size() << "k " << k << "|G'(V)|^k " << NumberOfVerticesToThePowerOfK << std::endl;
    /* Iterate through all k-combinations of vertices */
    int u,v;
    for (int x = 0; x < NumberOfVerticesToThePowerOfK; ++x){
        for (int z = 0; z < k; ++z){
            u = combinations[x*k + z];
            for (int i = csr->row_offsets[u]; i < csr->row_offsets[u+1]; ++i){
                v = csr->column_indices[i];
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
    for (int x = 0; x < NumberOfVerticesToThePowerOfK; ++x){
        vectorOfSetsOfEdgesCoveredByBuss[x].insert(g.edgesCoveredByKernelization.begin(), g.edgesCoveredByKernelization.end());
        if(vectorOfSetsOfEdgesCoveredByBuss[x].size() - totalEdgeCount == 0)
            results[x] = 1;
        else
            results[x] = 0;
    }
}

void SequentialBuss::PrintVCSets(){
    bool anyAnswerExists = false;
    for (int x = 0; x < NumberOfVerticesToThePowerOfK; ++x){
        if(results[x] != 0){
            anyAnswerExists = true;
            for (int z = 0; z < k; ++z){
                std::cout << " " << combinations[x*k + z];
            }
            std::cout << std::endl;
        }
    }
    if (!anyAnswerExists){
        std::cout << "No k-VC Found" << std::endl;
    } 
}


/* http://rosettacode.org/wiki/Combinations#C.2B.2B */
void SequentialBuss::PopulateCombinations(int * combinations_arg, int N, int K){
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's
    int rowI = 0;
    int colI = 0;
    // print integers and permute bitmask
    do {
        for (int i = 0; i < N; ++i) // [0..N-1] integers
        {
            if (bitmask[i]){
                /* Row length is K */
                combinations_arg[rowI*K + colI] = i;
                std::cout << " " << i;
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