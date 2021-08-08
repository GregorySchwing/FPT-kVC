#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <vector>
#include "COO.h"
#include "CSR.h"
#include <set>
#include <cstdlib> /* rand */
#include  <random>
#include  <iterator>
#include "ConnectednessTest.h"
#include "../common/CSVRange.h"

class Graph {
    public:
       // Graph(int vertexCount);
       // Graph(std::string filename, char sep = ',', int vertexCount = 0);
Graph(CSR & csr);
Graph(const Graph & other);
Graph(Graph & g_arg);
void Init(Graph & g_parent, std::vector<int> & verticesToDelete);
void SetMyOldsToParentsNews(Graph & g_parent);

/* Constructor to make induced subgraph G'' for each branch */
     //   Graph(Graph * g_arg, std::vector<int> & verticesToDelete);
        std::vector<int> & GetRemainingVerticesRef();
        std::vector<int> & GetNewDegRef();
        void SetOldDegRef(std::vector<int> & parents_new_ref);

        int GetEdgesLeftToCover();
        int GetDegree(int v);
        int GetOutgoingEdge(int v, int outEdgeIndex);
        
        int GetRandomOutgoingEdge(int v, std::vector<int> & path);
        void DFS(std::vector<int> & path, int rootVertex);


        CSR & GetCSR();
        COO * GetCOO();
        int edgesLeftToCover;
        std::set<std::pair<int,int>> edgesCoveredByKernelization;
        void removeVertex(int vertexToRemove, std::vector<int> & verticesRemaining);
        void PrepareGPrime(std::vector<int> & verticesToRemoveRef);
        int GetVertexCount();
        std::vector<int> & GetCondensedNewValRef();
        void PrintEdgesOfS();
        bool GPrimeEdgesGreaterKTimesKPrime(int k, int kPrime);
        int GetRandomVertex();


    private:
        int vertexCount;
        CSR csr;

        //CSR & csr_ref;
        // Every vertex touched by an edge removed should be checked after for being degree 0
        // and then removed if so, clearly the vertices chosen by the algorithm for removing
        // are also removed
        std::vector<int> verticesRemaining, vertexTouchedByRemovedEdge;
        
        // Following the CSR design pattern, a reference to the old degrees
        std::vector<int> & old_degrees_ref;
        // Following the CSR design pattern, the new degrees
        std::vector<int> new_degrees;

        std::vector<int> newValues;

        void BuildTheExampleCOO(COO * coordinateFormat);
        void BuildCOOFromFile(COO * coordinateFormat, std::string filename);
        void ProcessGraph(int vertexCount);
        void SetVertexCountFromEdges(COO * coordinateFormat);
        void SetEdgesOfSSymParallel(std::vector<int> & S);
        void SetEdgesLeftToCoverParallel();
        void SetNewRowOffsets(std::vector<int> & newRowOffsetsRef);
        void CountingSortParallelRowwiseValues(
                int procID,
                int beginIndex,
                int endIndex,
                std::vector<int> & A_row_offsets,
                std::vector<int> & A_column_indices,
                std::vector<int> & A_values,
                std::vector<int> & B_row_indices_ref,
                std::vector<int> & B_column_indices_ref,
                std::vector<int> & B_values_ref);
        void RemoveNewlyDegreeZeroVertices(     std::vector<int> & verticesToRemove,
                                                std::vector<int> & oldRowOffsets, 
                                                std::vector<int> & oldColumnIndices, 
                                                std::vector<int> & newRowOffsets);

};
#endif
