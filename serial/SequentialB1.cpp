#include "SequentialB1.h"

SequentialB1::SequentialB1( Graph & g_arg, 
                            int k_prime_arg):
                            g(g_arg), k_prime(k_prime_arg){

    std::vector<int> path;
    int randomVertex = g.GetRandomVertex();
    path.push_back(randomVertex);
    DFS(path, randomVertex);
    for (auto & v : path)
        std::cout << v << " ";
    std::cout << std::endl;
}

/* DFS of maximum length 3. No simple cycles u -> v -> u */
void SequentialB1::DFS(std::vector<int> & path, int rootVertex){
    if (path.size() == 4)
        return;

    int randomOutgoingEdge = g.GetRandomOutgoingEdge(rootVertex, path);
    if (randomOutgoingEdge < 0) {
        return;
    } else {
        path.push_back(randomOutgoingEdge);
        return DFS(path, randomOutgoingEdge);
    }
}

int SequentialB1::classifyPath(std::vector<int> & path){
    if (path.size()==2)
        return 3;
    else if (path.size()==3)
        return 2;
    else if (path.front() == path.back())
        return 1;
    else
        return 0;
}
