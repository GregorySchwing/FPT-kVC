#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <vector>
#include "COO.h"
#include "CSR.h"
#include "DegreeController.h"
#include "NeighborsBinaryDataStructure.h"
#include <set>

#include "RandomSampler.h"

class Graph {
    public:
        Graph(int vertexCount);
        Graph(Graph & g_arg);

        DegreeController * GetDegreeController();
        int GetEdgesLeftToCover();
        int GetRandomVertex();
        int GetDegree(int v);
        CSR * GetCSR();
        COO * GetCOO();
        void UpdateNeighBits();
        int edgesLeftToCover;
        std::set<std::pair<int,int>> edgesCoveredByKernelization;

    private:
        RandomSampler rs;
        COO * coordinateFormat;
        CSR * compressedSparseMatrix;
        DegreeController * degCont;
        NeighborsBinaryDataStructure * neighBits;

};
#endif
