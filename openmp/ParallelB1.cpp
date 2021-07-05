#include "ParallelB1.h"

ParallelB1::ParallelB1( Graph * g_arg,
                        int k_arg,
                        std::vector<int> & verticesToRemove_arg,
                        //std::vector<int> verticesRemaining_arg,
                        ParallelB1 * parent_arg):
                        g(g_arg),
                        k(k_arg), 
                        //verticesToRemove(verticesToRemove_arg),
                        //verticesRemainingRef()
                        parent(parent_arg),
                        result(false){
    verticesToRemove = verticesToRemove_arg;
    //verticesRemaining = verticesRemaining_arg;
    
    if (parent_arg == NULL){
        g->PrintEdgesOfS();   
        std::cout << g->edgesLeftToCover << " edges left in induced subgraph G'" << std::endl;
        std::cout << "verticesToRemove.size() " << verticesToRemove.size() << std::endl;
        //k = k_arg - verticesToRemove.size();
        std::cout << "Setting k' = k - b = " << k - verticesToRemove.size() << std::endl;
        bool noSolutionExists = g->GPrimeEdgesGreaterKTimesKPrime(k, k - verticesToRemove.size());
        if(noSolutionExists){
            std::cout << "|G'(E)| > k*k', no solution exists" << std::endl;
            return;
        } else{
            std::cout << "|G'(E)| <= k*k', a solution may exist" << std::endl;
        }
        g->PrepareGPrime(verticesToRemove);
        std::cout << "g->compressedSparseMatrix->new_row_offsets.size()" << g->GetCSR().GetNewRowOffRef().size() << std::endl;
        std::cout << "g->compressedSparseMatrix->new_column_indices.size()" << g->GetCSR().GetNewColRef().size() << std::endl;
        std::cout << "g->compressedSparseMatrix->new_values.size()" << g->GetCSR().GetNewValRef().size() << std::endl;

        std::vector<int> emptyVector;
        childrensVertices.resize(1);
        // Pointers to the children 
        children = new ParallelB1*[1];
        children[0] = new ParallelB1(new Graph(g,emptyVector),
                                    k - verticesToRemove.size(), 
                                    emptyVector,
                                    this); 
        return;
    }

    if(g_arg->edgesLeftToCover == 0){
        std::cout << "Found an answer" << std::endl;
        result = true;
        return;
    } else {
        std::cout << "Prepping G Prime" << std::endl;
        g->PrepareGPrime(verticesToRemove);
    }

    std::vector<int> path;
    std::cout << "Grabbing a randomVertex: " <<  std::endl;
    int randomVertex = g->GetRandomVertex();
    std::cout << "randomVertex: " << randomVertex << std::endl;

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
            children[i] = new ParallelB1(new Graph(g, childrensVertices[i]),
                                        k - childrensVertices[i].size(), 
                                        childrensVertices[i], 
                                        this);

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


void ParallelB1::IterateTreeStructure(ParallelB1 * root){
    std::cout << "Called IterateTreeStructure" << std::endl;
    ParallelB1 * next;
    if(root->GetResult()){
        std::cout << "Found an answer" << std::endl;
        std::vector<int> answer;
        TraverseUpTree(root, answer);
        std::cout << "Printing answer :" << std::endl;
        for (auto & v : answer)
            std::cout << v << " ";
        std::cout << std::endl;
    }
    for (int i = 0; i < root->GetNumberChildren(); ++i){
        next = root->GetChild(i);
        // If a child would have too many vertices, we just set it NULL
        if(next != NULL)
            IterateTreeStructure(next);
    }
}

void ParallelB1::TraverseUpTree(ParallelB1 * leaf, std::vector<int> & answer){
    for (auto & v : leaf->GetVerticesToRemove())
        answer.push_back(v);
    if(leaf->GetParent()!=NULL){
        TraverseUpTree(leaf->GetParent(), answer);
    }
}

ParallelB1 * ParallelB1::GetParent(){
    return parent;
}


int ParallelB1::GetNumberChildren(){
    return childrensVertices.size();
}

std::vector<int> ParallelB1::GetVerticesToRemove(){
    return verticesToRemove;
}

ParallelB1 * ParallelB1::GetChild(int i){
    return children[i];
}

bool ParallelB1::GetResult(){
    return result;
}