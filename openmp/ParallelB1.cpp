#include "SequentialB1.h"

SequentialB1::SequentialB1( Graph * g_arg, 
                            std::vector<int> verticesToRemove_arg,
                            int k_prime_arg,
                            SequentialB1 * parent_arg):
                            g(g_arg), 
                            verticesToRemove(verticesToRemove_arg),
                            k_prime(k_prime_arg), 
                            parent(parent_arg),
                            result(false){
    if (k_prime_arg <= 0 && g->edgesLeftToCover > 0){
        result = false;
        return;
    }

    if(k_prime_arg >= 0 && g->edgesLeftToCover == 0){
        result = true;
        return;
    }
    
    std::vector<int> path;
    int randomVertex = g->GetRandomVertex();
    path.push_back(randomVertex);
    DFS(path, randomVertex);
    for (auto & v : path)
        std::cout << v << " ";
    std::cout << std::endl;
    int caseNumber = classifyPath(path);
    std::cout << "Case number: " << caseNumber << std::endl;
    createVertexSetsForEachChild(caseNumber, path);

    /* Pointers to the children */
    children = new SequentialB1*[childrensVertices.size()];
    for (int i = 0; i < childrensVertices.size(); ++i){
        if (k_prime - childrensVertices[i].size() >= 0){
            std::cout << "Child " << i << std::endl;
            std::cout << "Printing Children Verts" << std::endl;
            for (auto & v : childrensVertices[i])
                std::cout << v << " ";
            std::cout << std::endl;
            std::cout << "Printed Children Verts" << std::endl;
            children[i] = new SequentialB1(new Graph(*g, childrensVertices[i]), childrensVertices[i], k_prime - childrensVertices[i].size(), this);
        } else{
            std::cout << "Child " << i << " is null" << std::endl;
            children[i] = NULL;
        }
    }
}

void SequentialB1::IterateTreeStructure(SequentialB1 * root){
    SequentialB1 * next;
    if(root->GetResult()){
        std::cout << "Found an answer" << std::endl;
        std::vector<int> answer;
        TraverseUpTree(root, answer);
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

void SequentialB1::TraverseUpTree(SequentialB1 * leaf, std::vector<int> & answer){
    for (auto & v : leaf->GetVerticesToRemove())
        answer.push_back(v);
    if(leaf->GetParent()!=NULL){
        TraverseUpTree(leaf->GetParent(), answer);
    }
}

SequentialB1 * SequentialB1::GetParent(){
    return parent;
}


int SequentialB1::GetNumberChildren(){
    return childrensVertices.size();
}

std::vector<int> SequentialB1::GetVerticesToRemove(){
    return verticesToRemove;
}

SequentialB1 * SequentialB1::GetChild(int i){
    return children[i];
}

bool SequentialB1::GetResult(){
    return result;
}


/* DFS of maximum length 3. No simple cycles u -> v -> u */
void SequentialB1::DFS(std::vector<int> & path, int rootVertex){
    if (path.size() == 4)
        return;

    int randomOutgoingEdge = g->GetRandomOutgoingEdge(rootVertex, path);
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

void SequentialB1::createVertexSetsForEachChild(int caseNumber, std::vector<int> & path){
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
