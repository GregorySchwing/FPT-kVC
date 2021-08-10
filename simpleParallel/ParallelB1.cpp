#include "ParallelB1.h"
#include <math.h>       /* pow */
// Sum of i = 0 to n/2
// 3^i
int ParallelB1::CalculateWorstCaseSpaceComplexity(int vertexCount){
    int summand= 0;
    // ceiling(vertexCount/2) loops
    for (int i = 0; i < (vertexCount + 2 - 1)/2; ++i)
        summand += pow (3.0, i);
    return summand;
}

void ParallelB1::PopulateTree(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer){
    // ceiling(vertexCount/2) loops
    int result, childVertex;
    for (int i = 0; i < treeSize; ++i){
        result = GenerateChildren(graphs[i]);
        while (graphs[i].GetChildrenVertices().size() == 1){
            graphs[i].InitGPrime(graphs[i], graphs[i].GetChildrenVertices().front());
            graphs[i].GetChildrenVertices().clear();
            result = GenerateChildren(graphs[i]);
        }        
        if (result == -1){
            TraverseUpTree(i, graphs, answer);
            return;
        } else {
            for (int c = 1; c <= 3; ++c)
                graphs[3*i + c].InitGPrime(graphs[i], graphs[i].GetChildrenVertices()[c-1]);
        }
    }
}


int ParallelB1::GenerateChildren(Graph & child_g){

    std::vector< std::vector<int> > & childrensVertices_ref = child_g.GetChildrenVertices();

    std::vector<int> path;
    int randomVertex = GetRandomVertex(child_g.GetRemainingVerticesRef());
    std::cout << "Grabbing a randomVertex: " <<  randomVertex<< std::endl;
    if(randomVertex == -1)
        return randomVertex;

    path.push_back(randomVertex);

    DFS(child_g.GetCSR().GetNewRowOffRef(), 
        child_g.GetCSR().GetNewColRef(), 
        path, 
        randomVertex);

    for (auto & v : path)
        std::cout << v << " ";
    std::cout << std::endl;
    int caseNumber = classifyPath(path);
    std::cout << "Case number: " << caseNumber << std::endl;
    createVertexSetsForEachChild(childrensVertices_ref, caseNumber, path);

    return 0;
}

int ParallelB1::GetRandomVertex(std::vector<int> & verticesRemaining){
    if(verticesRemaining.size() == 0)
        return -1;
    int index = rand() % verticesRemaining.size();
    return verticesRemaining[index];
}


/* DFS of maximum length 3. No simple cycles u -> v -> u */
void ParallelB1::DFS(std::vector<int> & new_row_off,
                    std::vector<int> & new_col_ref, 
                    std::vector<int> & path, 
                    int rootVertex){
    if (path.size() == 4)
        return;

    int randomOutgoingEdge = GetRandomOutgoingEdge(new_row_off, new_col_ref, rootVertex, path);
    if (randomOutgoingEdge < 0) {
        return;
    } else {
        path.push_back(randomOutgoingEdge);
        return DFS(new_row_off, new_col_ref, path, randomOutgoingEdge);
    }
}

int ParallelB1::GetRandomOutgoingEdge(  std::vector<int> & new_row_off,
                                        std::vector<int> & new_col_ref,
                                        int v, 
                                        std::vector<int> & path){

    std::vector<int> outgoingEdges(&new_col_ref[new_row_off[v]],
                        &new_col_ref[new_row_off[v+1]]);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(outgoingEdges.begin(), outgoingEdges.end(), g);
    std::vector<int>::iterator it = outgoingEdges.begin();

    while (it != outgoingEdges.end()){
        /* To prevent simple paths, must at least have 2 entries, 
        assuming there are no self edges, since the first entry, v,
        is randomly chosen and the second entry is a random out edge */
        if (path.size() > 1 && *it == path.rbegin()[1]) {
            //std::cout << "Wouldve been a simple path, skipping " << *it << std::endl;
            ++it;
        } else
            return *it;
    }

    return -1;
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

void ParallelB1::createVertexSetsForEachChild(std::vector< std::vector<int> > & childrensVertices,
                                            int caseNumber, 
                                            std::vector<int> & path){
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

void ParallelB1::TraverseUpTree(int index, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer){
    bool haventReachedRoot = true;
    while(haventReachedRoot) {
        if (index == 0)
            haventReachedRoot = false;
        for (auto & v : graphs[index].GetVerticesThisGraphIncludedInTheCover())
            answer.push_back(v);
        index = (index-1)/3;
    } 
}

