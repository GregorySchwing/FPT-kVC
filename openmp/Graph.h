#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <vector>
#include "COO.h"
#include "CSR.h"
#include "DegreeController.h"
#include "NeighborsBinaryDataStructure.h"
#include <set>

class Graph {
    public:
        Graph(int vertexCount);
        Graph(Graph & g_arg);
/* Constructor to make induced subgraph G'' for each branch */
        Graph(CSR * csr_arg, std::vector<int> & verticesToDelete);
        std::vector<int> & GetRemainingVerticesRef();
        int GetEdgesLeftToCover();
        int GetDegree(int v);
        int GetOutgoingEdge(int v, int outEdgeIndex);
        int GetRandomOutgoingEdge(int v, std::vector<int> & path);
        CSR * GetCSR();
        COO * GetCOO();
        int edgesLeftToCover;
        std::set<std::pair<int,int>> edgesCoveredByKernelization;
        void removeVertex(int vertexToRemove, std::vector<int> & verticesRemaining);
        void PrepareGPrime();
        int GetNumberOfRows();
        std::vector<int> & GetCondensedNewValRef();


    private:
        int numberOfRows;
        COO * coordinateFormat;
        CSR * compressedSparseMatrix;
        // Every vertex touched by an edge removed should be checked after for being degree 0
        // and then removed if so, clearly the vertices chosen by the algorithm for removing
        // are also removed
        std::vector<int> verticesRemaining, newDegrees, newValues, vertexTouchedByRemovedEdge;

        //std::vector<int> newRowOffsets, newColumnIndices, newValues, ;

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
        void RemoveDegreeZeroVertices(std::vector<int> & newRowOffsets);

};
#endif
