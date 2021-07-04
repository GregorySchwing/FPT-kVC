#include "ParallelB1.h"

ParallelB1::ParallelB1( Graph * g_arg,
                        int k_arg,
                        std::vector<int> & verticesToRemove_arg,
                        //std::vector<int> verticesRemaining_arg,
                        ParallelB1 * parent_arg):
                        g(g_arg),
                        k(k_arg), 
                        verticesToRemoveRef(verticesToRemove_arg),
                        //verticesRemainingRef()
                        parent(parent_arg),
                        result(false){

    //verticesRemaining = verticesRemaining_arg;
    
    if (parent_arg == NULL){
        g->PrintEdgesOfS();   
        std::cout << g->edgesLeftToCover << " edges left in induced subgraph G'" << std::endl;
        std::cout << "verticesToRemove.size() " << verticesToRemoveRef.size() << std::endl;
        //k = k_arg - verticesToRemove.size();
        std::cout << "Setting k' = k - b = " << k - verticesToRemoveRef.size() << std::endl;
        bool noSolutionExists = g->GPrimeEdgesGreaterKTimesKPrime(k, k - verticesToRemoveRef.size());
        if(noSolutionExists){
            std::cout << "|G'(E)| > k*k', no solution exists" << std::endl;
            return;
        } else{
            std::cout << "|G'(E)| <= k*k', a solution may exist" << std::endl;
        }
        g->PrepareGPrime();
        std::vector<int> emptyVector;
        // Pointers to the children 
        children = new ParallelB1*[1];
        children[0] = new ParallelB1(g,
                                    k - verticesToRemoveRef.size(), 
                                    emptyVector,
                                    this);
        return;
    }

    if(g_arg->edgesLeftToCover == 0){
        result = true;
        return;
    } else {
        g->PrepareGPrime();
    }

    std::vector<int> path;
    int randomVertex = g->GetRandomVertex();
    path.push_back(randomVertex);
    g->DFS(path, randomVertex);
    for (auto & v : path)
        std::cout << v << " ";
    std::cout << std::endl;
    int caseNumber = classifyPath(path);
    std::cout << "Case number: " << caseNumber << std::endl;
    createVertexSetsForEachChild(caseNumber, path);
    
    // Pointers to the children 
    children = new ParallelB1*[childrensVertices.size()];
    for (int i = 0; i < childrensVertices.size(); ++i){
        if (k - childrensVertices[i].size() >= 0){
            std::cout << "Child " << i << std::endl;
            std::cout << "Printing Children Verts" << std::endl;
            for (auto & v : childrensVertices[i])
                std::cout << v << " ";
            std::cout << std::endl;
            std::cout << "Printed Children Verts" << std::endl;
/*
            CSR * new_csr = new CSR(g->GetNumberOfRows(),
                    g->GetCSR()->GetNewRowOffRef(), 
                    g->GetCSR()->GetNewColRef(), 
                    g->GetCondensedNewValRef());
            children[i] = new ParallelB1(new Graph(g, childrensVertices[i]),
                                        k - childrensVertices[i].size(), 
                                        childrensVertices[i], 
                                        this,
);
*/
        } else{
            std::cout << "Child " << i << " is null" << std::endl;
            children[i] = NULL;
        }
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

