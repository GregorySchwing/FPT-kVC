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
//Graph(Graph & g_arg);
// Since for GPrime, the vertices to include in the cover, S,
// come from the kernelization step
void InitGPrime(Graph & g_parent, 
                std::vector<int> & S);
void InitGNPrime(Graph & g_parent, 
                std::vector<int> & verticesToIncludeInCover);
void SetMyOldsToParentsNews(Graph & g_parent);
void ClearNewDegrees(Graph & g_parent);
void PopulatePreallocatedMemory(Graph & g_parent);
std::vector<int> & GetVerticesThisGraphIncludedInTheCover();

/* Constructor to make induced subgraph G'' for each branch */
     //   Graph(Graph * g_arg, std::vector<int> & verticesToDelete);
        std::vector<int> & GetRemainingVerticesRef();
         std::vector<int> & GetNewDegRef();
         std::vector<int> * GetOldDegPointer();
         std::vector<int> & GetOldDegRef();


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
        void InduceSubgraph(std::vector<int> & verticesToRemoveRef);
        int GetVertexCount();
        void PrintEdgesOfS();
        bool GPrimeEdgesGreaterKTimesKPrime(int k, int kPrime);
        std::vector< std::vector<int> > & GetChildrenVertices();
        void SetVerticesToIncludeInCover(std::vector<int> & verticesRef);

    private:
        Graph * parent;
        std::vector<int> verticesToIncludeInCover;
        int vertexCount;
        CSR csr;

        std::vector < std::vector<int> > childrenVertices;        
        // Every vertex touched by an edge removed should be checked after for being degree 0
        // and then removed if so, clearly the vertices chosen by the algorithm for removing
        // are also removed
        std::vector<int> verticesRemaining, vertexTouchedByRemovedEdge;
        
        // Following the CSR design pattern, a reference to the old degrees
        // For Original G, this comes from the ParallelKernel class
        std::vector<int> * old_degrees_ref;
        // Only used in first graph.
        std::vector<int> old_degrees;

        // Following the CSR design pattern, the new degrees
        std::vector<int> new_degrees;


        void BuildTheExampleCOO(COO * coordinateFormat);
        void BuildCOOFromFile(COO * coordinateFormat, std::string filename);
        void ProcessGraph(int vertexCount);
        void SetVertexCountFromEdges(COO * coordinateFormat);
        void SetEdgesOfSSymParallel(std::vector<int> & S);
        void SetEdgesLeftToCoverParallel();
        void SetNewRowOffsets(std::vector<int> & newRowOffsetsRef);
        void SetOldDegRef(std::vector<int> & old_deg_ref);
        void SetParent(Graph & g_parent);

        void CalculateNewRowOffsets();
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
