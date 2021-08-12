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
    void ProcessImmediately(std::vector<int> & S);


    std::vector<int> & GetVerticesThisGraphIncludedInTheCover();

    std::vector<int> GetRemainingVertices();
    std::vector<int> & GetRemainingVerticesRef();
    std::vector<int> GetHasntBeenRemoved();

    void SetRemainingVertices(std::vector<int> verticesRemaining_arg);
    void SetHasntBeenRemoved(std::vector<int> hasntBeenRemoved_arg);

    std::vector<int> & GetNewDegRef();
    std::vector<int> * GetOldDegPointer();
    std::vector<int> & GetOldDegRef();


    int GetEdgesLeftToCover();
    int GetDegree(int v);


    CSR & GetCSR();
    COO * GetCOO();
    std::set<std::pair<int,int>> edgesCoveredByKernelization;
    void removeVertex(int vertexToRemove);
    void InduceSubgraph();
    int GetVertexCount();
    void PrintEdgesOfS();
    void PrintEdgesRemaining();
    bool GPrimeEdgesGreaterKTimesKPrime(int k, int kPrime);
    std::vector< std::vector<int> > & GetChildrenVertices();
    void SetVerticesToIncludeInCover(std::vector<int> & verticesRef);
    Graph & GetParent();


private:
    Graph * parent;
    int vertexCount;
    CSR csr;
    std::vector<int> testVals;
    // vector of vectors of the children
    std::vector < std::vector<int> > childrenVertices;
    // The vertices passed as an argument to the InitGPrime method, used for creating an answer
    std::vector<int> verticesToIncludeInCover;
    // set of vertices remaining, removed as vertices become degree 0
    std::vector<int> verticesRemaining;
    // array of length vertexCount of booleans
    std::vector<int> hasntBeenRemoved;
    // Set by SetEdgesLeftToCoverParallel method
    int edgesLeftToCover;
    
    // Following the CSR design pattern, a reference to the old degrees
    // For Original G, this comes from the ParallelKernel class
    std::vector<int> * old_degrees_ref;
    // Only used in first Graph
    std::vector<int> old_degrees;


    // Following the CSR design pattern, the new degrees
    std::vector<int> new_degrees;

    void ProcessGraph(int vertexCount);

    void SetMyOldsToParentsNews(Graph & g_parent);
    void SetVerticesRemainingAndVerticesRemoved(Graph & g_parent);
    void PopulatePreallocatedMemory(Graph & g_parent);
    void PopulatePreallocatedMemoryFirstGraph(Graph & g_parent);

    void SetVertexCountFromEdges(COO * coordinateFormat);
    void SetEdgesOfSSymParallel(std::vector<int> & S,
                                std::vector<int> & row_offsets_ref,
                                std::vector<int> & column_indices_ref);
    void SetEdgesLeftToCoverParallel(std::vector<int> & row_offsets);
    void SetNewRowOffsets(std::vector<int> & newRowOffsetsRef);
    void SetOldDegRef(std::vector<int> & old_deg_ref);
    void SetParent(Graph & g_parent);

    void CalculateNewRowOffsets(std::vector<int> & old_degrees);
    void CountingSortParallelRowwiseValues(
            int procID,
            int beginIndex,
            int endIndex,
            std::vector<int> & A_row_offsets,
            std::vector<int> & A_column_indices,
            std::vector<int> & A_values,
            std::vector<int> & B_row_indices_ref,
            std::vector<int> & B_column_indices_ref);
            //,
            //std::vector<int> & B_values_ref);

    void RemoveNewlyDegreeZeroVertices(std::vector<int> & verticesToRemove,
                                        std::vector<int> & oldRowOffets,
                                        std::vector<int> & oldColumnIndices);

};
#endif
