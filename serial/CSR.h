#ifndef CSR_H
#define CSR_H

#include "SparseMatrix.h"
#include "COO.h"
#include "../VectorUtilities.h"


class CSR : public SparseMatrix {
    public:
        CSR(const COO & c);
        CSR(const CSR & c);
        /* For post-kernelization G' induced subgraph */
        CSR(const CSR & c, int edgesLeftToCover);

        /* For branch owned G'' induced subgraph */
        CSR(const CSR & c, std::vector<int> & verticesToDelete);

        std::string toString();
        void removeVertexEdges(int u);
        void removeVertexEdges(int u, std::vector<int> & valuesToModify, const CSR & c);
        void insertElements(const SparseMatrix & s);
        const CSR& castSparseMatrix(const SparseMatrix & s);
        std::vector<int> column_indices, row_offsets;
};

#endif