#ifndef CSR_H
#define CSR_H

#include "SparseMatrix.h"
#include "COO.h"
#include "../VectorUtilities.h"


class CSR : public SparseMatrix {
    public:
        CSR(const COO & c);
        std::string toString();
        void insertElements(const SparseMatrix & s);
        const CSR& castSparseMatrix(const SparseMatrix & s);
        std::vector<int> column_indices, row_offsets;
};

#endif