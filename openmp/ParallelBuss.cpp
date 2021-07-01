#include "ParallelBuss.h"

/* Vertex Cover Using Buss’ algorithm */
ParallelBuss::ParallelBuss(Graph & g_arg, int k_arg, int k_prime_arg):g(g_arg), k(k_arg), k_prime(k_prime_arg){

    /* G(V) - Verts with degree > k = GPrimeVertices <= 2k^2 */
    SetGPrimeVertices();
    /* Hence, GPrimeVertices^k <= (2*k^2)^k */
    
    NumberOfGPrimeVerticesChooseKPrime = choose(verticesOfGPrime.size(), k_prime);
    //std::cout << "NumberOfGPrimeVerticesChooseKPrime " << NumberOfGPrimeVerticesChooseKPrime << std::endl;
    results = new int[NumberOfGPrimeVerticesChooseKPrime];

    /* Hence, the amount of space to store the NumberOfVertices^k combinations
        each of size k, is (NumberOfVertices^k)*k */
    //std::cout << "NumberOfGPrimeVerticesChooseKPrime*k_prime = " << NumberOfGPrimeVerticesChooseKPrime*k_prime << std::endl;
    combinations = new int[NumberOfGPrimeVerticesChooseKPrime*k_prime];
    PopulateCombinations(combinations, verticesOfGPrime, k_prime);

    /* The sets of edges covered by each of the NumberOfVertices^k combinations */
    vectorOfSetsOfEdgesCoveredByBuss.resize(NumberOfGPrimeVerticesChooseKPrime);
    GenerateEdgeSets();
    //PrintEdgeSets();
    UnionKernelEdgesAndBFSEdges();
}
ParallelBuss::~ParallelBuss(){
    delete[] results;
    delete[] combinations;
}

void ParallelBuss::SetGPrimeVertices(){
    /*
    std::vector< std::vector<int> > & tempDegCont = (g.GetDegreeController())->GetTempDegCont();
    std::vector< std::vector<int> >::const_iterator it = tempDegCont.cbegin();
    std::cout << "G'(V) = {";
    while(it != (tempDegCont.cbegin() + k + 1)){
        for (auto & e : *it){
            std::cout << e << " ";
            verticesOfGPrime.push_back(e);
        }
        it++;
    }
    std::cout << "}";*/
}
void ParallelBuss::GenerateEdgeSets(){
    //std::cout << "|G'(V)| " << verticesOfGPrime.size() << " k_prime " << k_prime << " |G'(V)| Choose k_prime " << NumberOfGPrimeVerticesChooseKPrime << std::endl;
    /* Iterate through all k_prime-combinations of vertices */
    int u,v;
    for (int x = 0; x < NumberOfGPrimeVerticesChooseKPrime; ++x){
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

void ParallelBuss::PrintEdgeSets(){
    for (int x = 0; x < NumberOfGPrimeVerticesChooseKPrime; ++x){
        for (std::set<std::pair<int,int>>::iterator it = vectorOfSetsOfEdgesCoveredByBuss[x].begin();
            it != vectorOfSetsOfEdgesCoveredByBuss[x].end();
                ++it){
            std::cout << "(" << it->first << ", " << it->second << "), ";
        }
        std::cout << std::endl;
    }
}

void ParallelBuss::UnionKernelEdgesAndBFSEdges(){
    int totalEdgeCount = g.GetCSR()->column_indices.size()/2;
    for (int x = 0; x < NumberOfGPrimeVerticesChooseKPrime; ++x){
        vectorOfSetsOfEdgesCoveredByBuss[x].insert(g.edgesCoveredByKernelization.begin(), g.edgesCoveredByKernelization.end());
        if(vectorOfSetsOfEdgesCoveredByBuss[x].size() - totalEdgeCount == 0)
            results[x] = 1;
        else
            results[x] = 0;
    }
}

void ParallelBuss::PrintVCSets(){
    bool anyAnswerExists = false;
    int VCCount = 0;
    for (int x = 0; x < NumberOfGPrimeVerticesChooseKPrime; ++x){
        if(results[x] != 0){
            std::cout << "VC #" << VCCount << " = {";
            anyAnswerExists = true;
            for (int z = 0; z < k_prime; ++z){
                std::cout << combinations[x*k_prime + z] << ", " ;
            }
            std::cout << "}" << std::endl;
            ++VCCount;
        }
    }
    if (!anyAnswerExists){
        std::cout << "No k-VC Found" << std::endl;
    } 
}


/* http://rosettacode.org/wiki/Combinations#C.2B.2B */
void ParallelBuss::PopulateCombinations(int * combinations_arg, std::vector<int> & gPrimeVertices, int k_prime){
    std::string bitmask(k_prime, 1); // k_prime leading 1's
    bitmask.resize(gPrimeVertices.size(), 0); // N-K trailing 0's
    int rowI = 0;
    int colI = 0;
    // print integers and permute bitmask
    int iter = 0;
    std::cout << std::endl;
    do {
        colI = 0;
        for (int i = 0; i < gPrimeVertices.size(); ++i) // [0..G'(V)] integers
        {
            if (bitmask[i]){
                /* Row length is k_prime */
                combinations_arg[rowI*k_prime + colI] = gPrimeVertices[i];
                //std::cout << " " << gPrimeVertices[i];
                ++colI;
            }
        }
        //std::cout << std::endl;
        ++rowI;
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
}

/* https://stackoverflow.com/questions/1505675/power-of-an-integer-in-c */
int ParallelBuss::myPow(int x, unsigned int p)
{
  if (p == 0) return 1;
  if (p == 1) return x;
  
  int tmp = myPow(x, p/2);
  if (p%2 == 0) return tmp * tmp;
    else return x * tmp * tmp;
}

int ParallelBuss::choose(int n, int k){
    if (k == 0) return 1;
    return (n * choose(n - 1, k - 1)) / k;
}