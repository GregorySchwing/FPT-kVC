#include "openmp/Graph.h"
#include "openmp/COO.h"
#include "openmp/CSR.h"
#include "openmp/ParallelKernelization.h"
#include "openmp/ParallelB1.h"
//#include "gpu/COO.cuh"


bool KCoverExists(  std::vector<int> & answer,
                    ParallelKernelization & sk,
                    Graph * g,
                    int k){
    bool exists;
    sk.TestAValueOfK(k);
    sk.PrintS();            
    Graph gPrime(g, sk.GetS()); 
    ParallelB1 pb1(&gPrime, 
                    k, 
                    sk.GetS());
    exists = pb1.IterateTreeStructure(&pb1, answer);
    if (exists) {
        std::cout << "solution found" << std::endl;
        std::cout << "Printing answer :" << std::endl;
        for (auto & v : answer)
            std::cout << v << " ";
        std::cout << std::endl;
    } else {
        std::cout << "No solution found" << std::endl;
    }
    return exists;
}

int binarySearch(std::vector<int> & answer, 
                    int minK, 
                    int maxK,
                    ParallelKernelization & sk,
                    Graph * g){
    int left = minK;
    int right = maxK;
    int m;
    while (left <= right){
        m = (left + right)/2;
        if (KCoverExists(answer, sk, g, m)){
            return m;
        } else {
            left = m + 1;
        }
    }
    return m;
}


int main(int argc, char *argv[])
{
    /* Num edges, N, N, random entries? */
/*    std::cout << "Building G" << std::endl;
    Graph g(10);
    int k = 4;
    std::cout << "Building PK" << std::endl;
    ParallelKernelization sk(g, k);
    Graph * gPrime = new Graph(&g, sk.GetS());
    ParallelB1 pb1(gPrime, 
                k, 
                sk.GetS());
                //, gPrime->GetRemainingVerticesRef());
    pb1.IterateTreeStructure(&pb1);
*/
    std::cout << "Building G" << std::endl;
    Graph g("0.edges");
    int k = 4;
    std::cout << "Building PK" << std::endl;
    ParallelKernelization sk(g, k);
    int minK = 0;
    int maxK = g.GetVertexCount();
    for (int i = k; i < g.GetVertexCount(); ++i){
        // If (noSolutionExists)
        // If (Also clears and sets S if a sol'n could exist)
        if (sk.TestAValueOfK(i))
            continue;
        else{
            minK = i;
            break;
        }
    }
    std::cout << "Found min K : " << minK << std::endl;
    std::vector<int> answer;
    int kCovSize = binarySearch(answer,
                                minK,
                                maxK,
                                sk,
                                &g);
}