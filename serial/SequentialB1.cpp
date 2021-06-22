#include "SequentialB1.h"

SequentialB1::SequentialB1( Graph & g_arg, 
                            int k_prime_arg):
                            g(g_arg), k_prime(k_prime_arg){

    int counter = 0;
    std::vector<int> path;
    int randomVertex = g.GetRandomVertex();
    path.push_back(randomVertex);
    std::cout << randomVertex << " ";

    DFS(path, randomVertex, counter);
    for (auto & v : path)
        std::cout << v << " ";
    std::cout << std::endl;
}

/* DFS of maximum length 3. No simple cycles u -> v -> u */
void SequentialB1::DFS(std::vector<int> & path, int rootVertex, int & counter){
    if (path.size() == 4)
        return;

    int randomOutgoingEdge = g.GetRandomOutgoingEdge(rootVertex, path);

    if (randomOutgoingEdge < 0) {
        return;
    } else {
        path.push_back(randomOutgoingEdge);
        ++counter;
        DFS(path, randomOutgoingEdge, counter);
    }
}