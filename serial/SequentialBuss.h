#ifndef SEQUENTIALBUSS_H
#define SEQUENTIALBUSS_H

/* Vertex Cover Using Buss’ algorithm */
#include "Graph.h"
#include <iostream>

/* For : http://rosettacode.org/wiki/Combinations#C.2B.2B */
#include <algorithm>
#include <string>
/* For : http://rosettacode.org/wiki/Combinations#C.2B.2B */
class SequentialBuss {
    public:
        SequentialBuss(Graph & g_arg, int k_arg, int k_prime_arg);
        ~SequentialBuss();
        void PrintVCSets();
        /* Thanks to our asymmetric CSR, the edges in these sets will be disjoint from
            the edges covered by the kernelization step. */
        std::vector< std::set<std::pair<int,int>> > vectorOfSetsOfEdgesCoveredByBuss;

    private:
        Graph & g;
        int k;
        int k_prime;
	    int NumberOfVerticesToThePowerOfKPrime;
        int * results;
        int * combinations;        
        std::vector<int> verticesOfGPrime;
        void SetGPrimeVertices();
        void GenerateEdgeSets();
        void UnionKernelEdgesAndBFSEdges();
		/* http://rosettacode.org/wiki/Combinations#C.2B.2B */
        void PopulateCombinations(int * combinations_arg, std::vector<int> & gPrimeVertices, int N, int K);
        /* https://stackoverflow.com/questions/1505675/power-of-an-integer-in-c */
        int myPow(int x, unsigned int p);
};
#endif