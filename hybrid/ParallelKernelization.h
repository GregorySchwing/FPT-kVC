#ifndef Parallel_Kernelization_H
#define Parallel_Kernelization_H

#include "Graph.h"
#include <iostream>
#include "omp.h"
#include "LinearTimeDegreeSort.h"
#include <thrust/host_vector.h>

/* This class applies the kernelization used in Cheetham. */

class ParallelKernelization {
    public:
        ParallelKernelization(Graph & g_arg, int k_arg);
        void ParallelRadixSortWrapper(int procID,
                int beginIndex,
                int endIndex,
                thrust::host_vector<int> & A_row_indices,
                thrust::host_vector<int> & A_column_indices,
                thrust::host_vector<int> & A_values,
                thrust::host_vector<int> & B_row_indices_ref,
                thrust::host_vector<int> & B_column_indices_ref,
                thrust::host_vector<int> & B_values_ref);
        int GetKPrime();
        
        thrust::host_vector<int> & GetS();

        bool noSolutionExists;
        /*
                // These are only called on the Kernel to B1 transition 

        int GetRandomVertex();
        thrust::host_vector<int> & GetRowOffRef();
        thrust::host_vector<int> & GetColRef();
        thrust::host_vector<int> & GetValRef();
        thrust::host_vector<int> & GetVerticesRemainingRef();*/
        int GetNumberOfRows();
        void PrintS();
        bool TestAValueOfK(int k_arg);
        bool EdgeCountKernel();
        Graph & GetGPrime();
    private:
        void ParallelRadixSortWorker(int procID,
                    int beginIndex,
                    int endIndex,
                    int digit,
                    int base,
                    thrust::host_vector<int> & A_row_indices,
                    thrust::host_vector<int> & A_column_indices,
                    thrust::host_vector<int> & A_values,
                    thrust::host_vector<int> & B_row_indices_ref,
                    thrust::host_vector<int> & B_column_indices_ref,
                    thrust::host_vector<int> & B_values_ref,
                    thrust::host_vector<int> & C_ref);


        void CountingSortParallel(
                        int procID,
                        int beginIndex,
                        int endIndex,
                        thrust::host_vector<int> & A_row_indices,
                        thrust::host_vector<int> & A_column_indices,
                        thrust::host_vector<int> & A_values,
                        thrust::host_vector<int> & B_row_indices_ref,
                        thrust::host_vector<int> & B_column_indices_ref,
                        thrust::host_vector<int> & B_values_ref);

        void CountingSortParallelRowwiseValues(
                int procID,
                int beginIndex,
                int endIndex,
                thrust::host_vector<int> & A_row_offsets,
                thrust::host_vector<int> & A_column_indices,
                thrust::host_vector<int> & A_values,
                thrust::host_vector<int> & B_row_indices_ref,
                thrust::host_vector<int> & B_column_indices_ref,
                thrust::host_vector<int> & B_values_ref);
        

        void RemoveSVertices();
        void RemoveDegreeZeroVertices();

        int numberOfElements, numberOfRows;

        //thrust::host_vector<int> degrees, newDegrees

        int kPrime;
        Graph & g;
        int k;
        int b;
        thrust::host_vector<int> S;
        // These are now in Graph like all other usages of G
        //thrust::host_vector<int> & row_offsets, & column_indices, & values, & verticesRemaining;
        //thrust::host_vector<int> newRowOffsets, newColumnIndices, newValues, vertexTouchedByRemovedEdge;
        
        // n, number of entries
        // A, B = [1 . . n]
        // C = [0 .. k], k = max(A)


        bool CardinalityOfSetDegreeGreaterK(thrust::host_vector<int> & degrees,
                                            thrust::host_vector<int> & vertexKeys);
        int GetSetOfVerticesDegreeGreaterK(thrust::host_vector<int> & degrees,
                                            thrust::host_vector<int> & vertexKeys);
        LinearTimeDegreeSort * ltds;
        thrust::host_vector<int> old_degrees;
        Graph gPrime;
        /*
        void PrintEdgesOfS();
        void SetEdgesOfSSym();
        void SetEdgesOfSSymParallel();
        void SetEdgesLeftToCover();
        void SetEdgesLeftToCoverParallel();
        void SetNewRowOffsets();
        bool GPrimeEdgesGreaterKTimesKPrime();
        */
};
#endif
