#ifndef CSR_H
#define CSR_H

#include "SparseMatrix.h"
#include "COO.h"
#include "../VectorUtilities.h"


class CSR : public SparseMatrix {
    public:
        CSR(COO & c);
        CSR(CSR & c);

        /* For post-kernelization G' induced subgraph */
        CSR(int numberOfRows,
            std::vector<int> & row_offsets_ref,
            std::vector<int> & column_indices_ref,
            std::vector<int> & values_ref);


        /* For branch owned G'' induced subgraph */
        CSR(int numberOfRows,
            CSR & c,
            std::vector<int> & values_ref_arg);

        std::vector<int> & GetOldRowOffRef();
        std::vector<int> & GetNewRowOffRef();
        std::vector<int> & GetOldColRef();
        std::vector<int> & GetNewColRef();
        std::vector<int> & GetOldValRef();
        std::vector<int> & GetNewValRef();

        std::string toString();
        // These are for the next CSR
        std::vector<int> column_indices, row_offsets;
        // These are the current CSR
        std::vector<int> & column_indices_ref, & row_offsets_ref;

};

#endif