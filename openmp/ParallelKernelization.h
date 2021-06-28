#ifndef Parallel_Kernelization_H
#define Parallel_Kernelization_H

#include "Graph.h"
#include <iostream>
#include "omp.h"
/* This class applies the kernelization used in Cheetham. */

class ParallelKernelization {
    public:
        ParallelKernelization(Graph & g_arg, int k_arg);
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
        bool noSolutionExists;
    private:
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
        
        
        int GetStartingIndexInA(int processorID);
        int GetEndingIndexInA(int processorID); 
        int GetBlockSize();

        int numberOfElements, numberOfProcessors, blockSize, rowBlockSize, numberOfRows;
        int kPrime;
        Graph & g;
        int k;
        int b;
        std::vector<int> S;

        // n, number of entries
        // A, B = [1 . . n]
        // C = [0 .. k], k = max(A)
        std::vector<int> B_row_indices;
        std::vector<int> B_column_indices; 
        std::vector<int> B_values; 
        std::vector<int> C;

        bool CardinalityOfSetDegreeGreaterK(std::vector<int> & degrees,
                                            std::vector<int> & vertexKeys);
        int GetSetOfVerticesDegreeGreaterK(int k, std::vector<int> & S, DegreeController * degCont);
        void PrintS();
        void PrintEdgesOfS();

        void SetEdgesOfS(CSR * csr);
        void SetEdgesOfSSym(CSR * csr);
        void SetEdgesLeftToCover();
        int GetCardinalityOfSEdges();
        bool GPrimeEdgesGreaterKTimesKPrime();
};
#endif
