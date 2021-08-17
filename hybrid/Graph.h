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
#include "../lib/CSVRange.h"

class Graph {
public:
    Graph(CSR & csr);
    /* Constructor to allocate induced subgraph G'' for each branch */
    Graph(const Graph & other);

    void InitG(Graph & g_parent, thrust::host_vector<int> & S);
    void InitGPrime(Graph & g_parent, 
                    thrust::host_vector<int> & S);
    void ProcessImmediately(thrust::host_vector<int> & S);


    thrust::host_vector<int> & GetVerticesThisGraphIncludedInTheCover();

    thrust::host_vector<int> GetRemainingVertices();
    thrust::host_vector<int> & GetRemainingVerticesRef();
    thrust::host_vector<int> GetHasntBeenRemoved();
    thrust::host_vector<int> & GetHasntBeenRemovedRef();

    void SetRemainingVertices(thrust::host_vector<int> & verticesRemaining_arg);
    void SetHasntBeenRemoved(thrust::host_vector<int> & hasntBeenRemoved_arg);

    thrust::host_vector<int> & GetNewDegRef();
    thrust::host_vector<int> * GetOldDegPointer();
    thrust::host_vector<int> & GetOldDegRef();

    int GetSize();
    int GetNumberOfRows();

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
    std::vector< thrust::host_vector<int> > & GetChildrenVertices();
    void SetVerticesToIncludeInCover(thrust::host_vector<int> & verticesRef);
    Graph & GetParent();


private:
    Graph * parent;
    int vertexCount;
    CSR csr;
    thrust::host_vector<int> testVals;
    // vector of vectors of the children
    std::vector < thrust::host_vector<int> > childrenVertices;
    // The vertices passed as an argument to the InitGPrime method, used for creating an answer
    thrust::host_vector<int> verticesToIncludeInCover;
    // set of vertices remaining, removed as vertices become degree 0
    thrust::host_vector<int> verticesRemaining;
    // array of length vertexCount of booleans
    thrust::host_vector<int> hasntBeenRemoved;
    // Set by SetEdgesLeftToCoverParallel method
    int edgesLeftToCover;
    
    // Following the CSR design pattern, a reference to the old degrees
    // For Original G, this comes from the ParallelKernel class
    thrust::host_vector<int> * old_degrees_ref;
    // Only used in first Graph
    thrust::host_vector<int> old_degrees;


    // Following the CSR design pattern, the new degrees
    thrust::host_vector<int> new_degrees;

    void ProcessGraph(int vertexCount);

    void SetMyOldsToParentsNews(Graph & g_parent);
    void SetVerticesRemainingAndVerticesRemoved(Graph & g_parent);
    void PopulatePreallocatedMemory(Graph & g_parent);
    void PopulatePreallocatedMemoryFirstGraph(Graph & g_parent);

    void SetVertexCountFromEdges(COO * coordinateFormat);
    void SetEdgesOfSSymParallel(thrust::host_vector<int> & S,
                                thrust::host_vector<int> & row_offsets_ref,
                                thrust::host_vector<int> & column_indices_ref);
    void SetEdgesLeftToCoverParallel(thrust::host_vector<int> & row_offsets);
    void SetNewRowOffsets(thrust::host_vector<int> & newRowOffsetsRef);
    void SetOldDegRef(thrust::host_vector<int> & old_deg_ref);
    void SetParent(Graph & g_parent);

    void CalculateNewRowOffsets(thrust::host_vector<int> & old_degrees);
    void CountingSortParallelRowwiseValues(
            int procID,
            int beginIndex,
            int endIndex,
            thrust::host_vector<int> & A_row_offsets,
            thrust::host_vector<int> & A_column_indices,
            thrust::host_vector<int> & A_values,
            thrust::host_vector<int> & B_row_indices_ref,
            thrust::host_vector<int> & B_column_indices_ref);
            //,
            //thrust::host_vector<int> & B_values_ref);

    void RemoveNewlyDegreeZeroVertices(thrust::host_vector<int> & verticesToRemove,
                                        thrust::host_vector<int> & oldRowOffets,
                                        thrust::host_vector<int> & oldColumnIndices);

    friend class Graph_GPU;

};
#endif
