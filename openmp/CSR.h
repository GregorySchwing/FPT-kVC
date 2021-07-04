#ifndef CSR_H
#define CSR_H

#include "SparseMatrix.h"
#include "COO.h"
#include "../VectorUtilities.h"


class CSR : public SparseMatrix {
    public:
        CSR(COO & c);
        CSR(CSR & c);

        std::vector<int> & GetOldRowOffRef();
        std::vector<int> & GetNewRowOffRef();
        std::vector<int> & GetOldColRef();
        std::vector<int> & GetNewColRef();
        std::vector<int> & GetOldValRef();
        std::vector<int> & GetNewValRef();

        std::string toString();
        // These are for the next CSR
        std::vector<int> new_column_indices, new_row_offsets;
        // These are the current CSR
        std::vector<int> & old_column_indices_ref, & old_row_offsets_ref;

};

#endif