#include "ParallelB1.h"

/* First call, vertices have already been removed */
ParallelB1::ParallelB1( int k_prime_arg,
                        ParallelKernelization * pk_arg,
                        ParallelB1 * parent_arg):
                        pk(pk_arg),
                        k_prime(k_prime_arg), 
                        parent(parent_arg),
                        result(false){
    
    std::vector<int> path;
    int randomVertex = pk->GetRandomVertex();
    path.push_back(randomVertex);

}

/* DFS of maximum length 3. No simple cycles u -> v -> u */
void ParallelB1::DFS(std::vector<int> & path, int rootVertex){
    if (path.size() == 4)
        return;

    int randomOutgoingEdge = pk->GetRandomOutgoingEdge(rootVertex, path);
    if (randomOutgoingEdge < 0) {
        return;
    } else {
        path.push_back(randomOutgoingEdge);
        return DFS(path, randomOutgoingEdge);
    }
}


