#ifndef Parallel_Kernelization_H
#define Parallel_Kernelization_H

#include "Graph.h"
#include <iostream>
#include "omp.h"
/* This class applies the kernelization used in Cheetham. */

class ParallelKernelization {
    public:
        ParallelKernelization(Graph & g_arg, int k_arg);
        int GetKPrime();
        std::vector<int> & GetS();
        bool noSolutionExists;
    private:
        int kPrime;
        Graph & g;
        int k;
        int b;
        std::vector<int> S;

        bool CardinalityOfSetDegreeGreaterK(DegreeController * degCont);
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
