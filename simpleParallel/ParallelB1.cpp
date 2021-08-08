#include "ParallelB1.h"

void ParallelB1::EdgeCountKernel( Graph & g_arg,
                            int k_arg,
                            std::vector<int> & verticesToRemove_arg,
                            Graph & parent_g){
        parent_g.Init(g_arg, verticesToRemove_arg);
        std::cout << g_arg.edgesLeftToCover << " edges left in induced subgraph G'" << std::endl;
        std::cout << "verticesToRemove_arg.size() " << verticesToRemove_arg.size() << std::endl;
        std::cout << "Setting k' = k - b = " << k_arg - verticesToRemove_arg.size() << std::endl;
        int kPrime = k_arg - verticesToRemove_arg.size();
        bool noSolutionExists = g_arg.GPrimeEdgesGreaterKTimesKPrime(k_arg, kPrime);
        if(noSolutionExists){
            std::cout << "|G'(E)| > k*k', no solution exists" << std::endl;
            return;
        } else{
            std::cout << "|G'(E)| <= k*k', a solution may exist" << std::endl;
        }
        /*
        g->PrepareGPrime(verticesToRemove_arg);
        std::cout << "g->compressedSparseMatrix->new_row_offsets.size()" << g->GetCSR().GetNewRowOffRef().size() << std::endl;
        std::cout << "g->compressedSparseMatrix->new_column_indices.size()" << g->GetCSR().GetNewColRef().size() << std::endl;
        std::cout << "g->compressedSparseMatrix->new_values.size()" << g->GetCSR().GetNewValRef().size() << std::endl;

        std::vector<int> emptyVector;
        childrensVertices.resize(1);
        // Pointers to the children 
        children = new ParallelB1*[1];
        children[0] = new ParallelB1(new Graph(g,emptyVector),
                                    k_arg - verticesToRemove_arg.size(), 
                                    emptyVector,
                                    this); 
        */
        return;
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

