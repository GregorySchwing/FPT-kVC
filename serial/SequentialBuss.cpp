#include "SequentialBuss.h"

/* Vertex Cover Using Buss’ algorithm */
SequentialBuss::SequentialBuss(int k_arg, Graph & g_arg):g(g_arg), k(k_arg){
    /* NumberOfVertices <= 2k^2 */
    /* Hence, NumberOfVertices^k <= (2*k^2)^k */
    NumberOfVerticesToThePowerOfK = myPow(g.GetVertexCount(), k);
    results = new int[NumberOfVerticesToThePowerOfK];

    /* Hence, the amount of space to store the NumberOfVertices^k combinations
        each of size k, is (NumberOfVertices^k)*k */
    combinations = new int[NumberOfVerticesToThePowerOfK*k];
    PopulateCombinations(combinations, g.GetVertexCount(), k);

}
SequentialBuss::~SequentialBuss(){
    delete[] results;
    delete[] combinations;
}
void SequentialBuss::FindCover(){
    bool edgeCovered = false;
    bool alledgesCoveredByKernelization = true;
    Vertex * vertices = g.GetVertices();
std::cout << "g.GetVertexCount()) " << g.GetVertexCount() << "k " << k<< "(g.GetVertexCount())^k " << NumberOfVerticesToThePowerOfK << std::endl;
    /* Iterate through all k-combinations of vertices */
    for (int x = 0; x < NumberOfVerticesToThePowerOfK; ++x){
        std::cout << "iteration " << x << std::endl;
        alledgesCoveredByKernelization = true;
        /* Currently edges belong to vertices, so the only way to iterate through all edges, is to iterate through all verts and then the edges of that vert */
        for (int y = 0; y < g.GetVertexCount(); ++y){
            std::cout << "vertex " << y << " Edges : " << std::endl;
            for (auto & e : vertices[y].GetEdges()){
                std::cout << "edge (" << y << ", " << e << ")"  << std::endl;
                edgeCovered = false;
                /* Check if any of the vertices in this combination cover this edge */
    /* y - the source, e - the destination (y,e) */
    /* We iterate over all vertices, but vertices removed from kernelization
        will have 0 edges, thus we don't modify alledgesCoveredByKernelization */
                std::cout << "combination {";
                for (int z = 0; z < k; ++z){
                    std::cout << z << ", ";
                    if (e == combinations[x*k + z] || y == combinations[x*k + z])
                        edgeCovered = true;
                }
                std::cout << "}" << std::endl;
                alledgesCoveredByKernelization &= edgeCovered; 
            }
        }
        results[x] = alledgesCoveredByKernelization;        
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