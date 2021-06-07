#ifndef Sequential_Kernelization_H
#define Sequential_Kernelization_H

#include "Graph.h"
#include <iostream>
#include <set>

class SequentialKernelization {
    public:
        SequentialKernelization(Graph & g_arg, int k_arg);
        
    private:
        int kPrime;
        Graph & g;
        int k;
        int b;
        std::vector<int> S;
        std::set<std::pair<int,int>> edges;

        bool CardinalityOfSetDegreeGreaterK(DegreeController * degCont);
        int GetSetOfVerticesDegreeGreaterK(int k, std::vector<int> & S, DegreeController * degCont);
        void PrintS();
        void PrintEdgesOfS();

        void SetEdgesOfS(CSR * csr);
        //void RemoveSFromG();
        int GetCardinalityOfSEdges();

        bool GPrimeEdgesGraterKTimesKPrime();
};
#endif
