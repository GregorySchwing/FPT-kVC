#ifndef NEIGHBORSBINARYDATASTRUCTURE_H
#define NEIGHBORSBINARYDATASTRUCTURE_H

#include "CSR.h"
#include <boost/dynamic_bitset.hpp>

class NeighborsBinaryDataStructure {
    public:

        NeighborsBinaryDataStructure(CSR * compressedSparseMatrix);
        void InducedSubgraphAdjMatrixAsym(CSR * compressedSparseMatrix);
        void InducedSubgraphAdjMatrixSym(CSR * compressedSparseMatrix);

        int GetDegree(int vertex);
        std::vector<boost::dynamic_bitset<> > twoDimensionalBitMatrixOfNeighboringVertices;

};

#endif