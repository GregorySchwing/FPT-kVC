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
    DFS(path, randomVertex);
    for (auto & v : path)
        std::cout << v << " ";
    std::cout << std::endl;
    int caseNumber = classifyPath(path);
    std::cout << "Case number: " << caseNumber << std::endl;
    createVertexSetsForEachChild(caseNumber, path);
    
    // Pointers to the children 
    children = new ParallelB1*[childrensVertices.size()];
    for (int i = 0; i < childrensVertices.size(); ++i){
        if (k_prime - childrensVertices[i].size() >= 0){
            std::cout << "Child " << i << std::endl;
            std::cout << "Printing Children Verts" << std::endl;
            for (auto & v : childrensVertices[i])
                std::cout << v << " ";
            std::cout << std::endl;
            std::cout << "Printed Children Verts" << std::endl;

            CSR * new_csr = new CSR(pk_arg->GetNumberOfRows(),
                                pk_arg->GetRowOffRef(), 
                                pk_arg->GetColRef(), 
                                pk_arg->GetValRef());
        
            children[i] = new ParallelB1(new Graph(new_csr, pk_arg->GetVerticesRemainingRef()),
                                        k_prime - childrensVertices[i].size(), 
                                        this,
                                        childrensVertices[i], 
                                        pk_arg->GetVerticesRemainingRef());
        } else{
            std::cout << "Child " << i << " is null" << std::endl;
            children[i] = NULL;
        }
    }
}

/* All calls after Kernel to B1 transition, vertices have NOT been removed */
ParallelB1::ParallelB1( Graph * g_arg,
                        int k_prime_arg,
                        ParallelB1 * parent_arg,
                        std::vector<int> & verticesToRemove,
                        std::vector<int> verticesRemaining
                        ):
                        k_prime(k_prime_arg), 
                        parent(parent_arg),
                        result(false){
    
    if(g_arg->edgesLeftToCover == 0){
        result = true;
        return;
    } else {
        g->PrepareGPrime();
    }

    std::vector<int> path;
    int randomVertex = GetRandomVertex(verticesRemaining);
    path.push_back(randomVertex);
    DFS(path, randomVertex);
    for (auto & v : path)
        std::cout << v << " ";
    std::cout << std::endl;
    int caseNumber = classifyPath(path);
    std::cout << "Case number: " << caseNumber << std::endl;
    createVertexSetsForEachChild(caseNumber, path);
    
    // Pointers to the children 
    children = new ParallelB1*[childrensVertices.size()];
    for (int i = 0; i < childrensVertices.size(); ++i){
        if (k_prime - childrensVertices[i].size() >= 0){
            std::cout << "Child " << i << std::endl;
            std::cout << "Printing Children Verts" << std::endl;
            for (auto & v : childrensVertices[i])
                std::cout << v << " ";
            std::cout << std::endl;
            std::cout << "Printed Children Verts" << std::endl;
            CSR * new_csr = new CSR(g->GetNumberOfRows(),
                    g->GetCSR()->GetNewRowOffRef(), 
                    g->GetCSR()->GetNewColRef(), 
                    g->GetCondensedNewValRef());
            children[i] = new ParallelB1(new Graph(new_csr, g->GetRemainingVerticesRef()),
                                        k_prime - childrensVertices[i].size(), 
                                        this,
                                        childrensVertices[i], 
                                        g->GetRemainingVerticesRef());
        } else{
            std::cout << "Child " << i << " is null" << std::endl;
            children[i] = NULL;
        }
    }
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

int ParallelB1::classifyPath(std::vector<int> & path){
    if (path.size()==2)
        return 3;
    else if (path.size()==3)
        return 2;
    else if (path.front() == path.back())
        return 1;
    else
        return 0;
}

void ParallelB1::createVertexSetsForEachChild(int caseNumber, std::vector<int> & path){
    if (caseNumber == 0) {
        /* 3 Children */
        childrensVertices.resize(3);
        /* Each with 2 vertices */
        for (auto & cV : childrensVertices)
            cV.reserve(2);
        childrensVertices[0].push_back(path[0]);
        childrensVertices[0].push_back(path[2]);

        childrensVertices[1].push_back(path[1]);
        childrensVertices[1].push_back(path[2]);

        childrensVertices[2].push_back(path[1]);
        childrensVertices[2].push_back(path[3]);

    } else if (caseNumber == 1) {

        /* 3 Children */
        childrensVertices.resize(3);
        /* Each with 2 vertices */
        for (auto & cV : childrensVertices)
            cV.reserve(2);
        childrensVertices[0].push_back(path[0]);
        childrensVertices[0].push_back(path[1]);

        childrensVertices[1].push_back(path[1]);
        childrensVertices[1].push_back(path[2]);

        childrensVertices[2].push_back(path[0]);
        childrensVertices[2].push_back(path[2]);

    } else if (caseNumber == 2) {

        childrensVertices.resize(1);
        childrensVertices[0].reserve(1);
        childrensVertices[0].push_back(path[1]);

    } else {

        childrensVertices.resize(1);
        childrensVertices[0].reserve(1);
        childrensVertices[0].push_back(path[0]);

    }
}

int ParallelB1::GetRandomVertex(std::vector<int> & verticesRemaining){
    int index = rand() % verticesRemaining.size();
    return verticesRemaining[index];
}