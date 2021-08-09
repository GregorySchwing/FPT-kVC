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
    Graph(CSR & csr);
    /* Constructor to allocate induced subgraph G'' for each branch */
    Graph(const Graph & other);

    void InitG(Graph & g_parent, std::vector<int> & S);
    void InitGPrime(Graph & g_parent, 
                    std::vector<int> & S);

    std::vector<int> & GetVerticesThisGraphIncludedInTheCover();

    std::vector<int> & GetRemainingVerticesRef();
    std::vector<int> & GetNewDegRef();
    std::vector<int> * GetOldDegPointer();
    std::vector<int> & GetOldDegRef();


    int GetEdgesLeftToCover();
    int GetDegree(int v);


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
    // Only used in first Graph
    std::vector<int> old_degrees;


    // Following the CSR design pattern, the new degrees
    std::vector<int> new_degrees;


    void BuildTheExampleCOO(COO * coordinateFormat);
    void BuildCOOFromFile(COO * coordinateFormat, std::string filename);
    void ProcessGraph(int vertexCount);

    void SetMyOldsToParentsNews(Graph & g_parent);
    void PopulatePreallocatedMemory(Graph & g_parent);
    void PopulatePreallocatedMemoryFirstGraph(Graph & g_parent);

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
