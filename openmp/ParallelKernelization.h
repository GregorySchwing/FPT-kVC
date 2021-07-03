#ifndef Parallel_Kernelization_H
#define Parallel_Kernelization_H

#include "Graph.h"
#include <iostream>
#include "omp.h"
/* This class applies the kernelization used in Cheetham. */

class ParallelKernelization {
    public:
        ParallelKernelization(Graph & g_arg, int k_arg);
        void ParallelRadixSortWrapper(int procID,
                int beginIndex,
                int endIndex,
                std::vector<int> & A_row_indices,
                std::vector<int> & A_column_indices,
                std::vector<int> & A_values,
                std::vector<int> & B_row_indices_ref,
                std::vector<int> & B_column_indices_ref,
                std::vector<int> & B_values_ref);
        int GetKPrime();
        std::vector<int> & GetS();
        /* These are only called on the Kernel to B1 transition */
        int GetRandomVertex();
        int GetRandomOutgoingEdge(int v, std::vector<int> & path);
        /* These are only called on the Kernel to B1 transition */
        bool noSolutionExists;
        std::vector<int> & GetRowOffRef();
        std::vector<int> & GetColRef();
        std::vector<int> & GetValRef();
        std::vector<int> & GetVerticesRemainingRef();
        int GetNumberOfRows();

    private:
        void ParallelRadixSortWorker(int procID,
                    int beginIndex,
                    int endIndex,
                    int digit,
                    int base,
                    std::vector<int> & A_row_indices,
                    std::vector<int> & A_column_indices,
                    std::vector<int> & A_values,
                    std::vector<int> & B_row_indices_ref,
                    std::vector<int> & B_column_indices_ref,
                    std::vector<int> & B_values_ref,
                    std::vector<int> & C_ref);
        void CountingSortSerial(int max,
                        std::vector<int> & A_row_indices,
                        std::vector<int> & A_column_indices,
                        std::vector<int> & A_values,
                        std::vector<int> & B_row_indices_ref,
                        std::vector<int> & B_column_indices_ref,
                        std::vector<int> & B_values_ref,
                        std::vector<int> & C_ref);

        void CountingSortParallel(
                        int procID,
                        int beginIndex,
                        int endIndex,
                        std::vector<int> & A_row_indices,
                        std::vector<int> & A_column_indices,
                        std::vector<int> & A_values,
                        std::vector<int> & B_row_indices_ref,
                        std::vector<int> & B_column_indices_ref,
                        std::vector<int> & B_values_ref);

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
        
        
        int GetStartingIndexInA(int processorID);
        int GetEndingIndexInA(int processorID); 
        int GetBlockSize();
        void RemoveSVertices();
        void RemoveDegreeZeroVertices();

        int numberOfElements, numberOfProcessors, blockSize, rowBlockSize, numberOfRows;
        int kPrime;
        Graph & g;
        int k;
        int b;
        std::vector<int> S;
        std::vector<int> & row_offsets, & column_indices, & values, & verticesRemaining;
        std::vector<int> newDegrees, newRowOffsets, newColumnIndices, newValues, vertexTouchedByRemovedEdge;
        // n, number of entries
        // A, B = [1 . . n]
        // C = [0 .. k], k = max(A)
        std::vector<int> B_row_indices;
        std::vector<int> B_column_indices; 
        std::vector<int> B_values; 
        std::vector<int> C;

        bool CardinalityOfSetDegreeGreaterK(std::vector<int> & degrees,
                                            std::vector<int> & vertexKeys);
        int GetSetOfVerticesDegreeGreaterK(std::vector<int> & degrees,
                                            std::vector<int> & vertexKeys);
        void PrintS();
        void PrintEdgesOfS();

        void SetEdgesOfSSym();
        void SetEdgesOfSSymParallel();
        void SetEdgesLeftToCover();
        void SetEdgesLeftToCoverParallel();
        void SetNewRowOffsets();
        bool GPrimeEdgesGreaterKTimesKPrime();

};
#endif
