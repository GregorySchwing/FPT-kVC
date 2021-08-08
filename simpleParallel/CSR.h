#ifndef CSR_H
#define CSR_H

#include "SparseMatrix.h"
#include "COO.h"
#include "../VectorUtilities.h"


class CSR : public SparseMatrix {
    public:
        CSR();
        CSR(COO & c);
        CSR(CSR & c);
        CSR(const CSR & c);


        std::vector<int> & GetOldRowOffRef();
        std::vector<int> & GetNewRowOffRef();
        std::vector<int> & GetOldColRef();
        std::vector<int> & GetNewColRef();
        std::vector<int> & GetOldValRef();
        std::vector<int> & GetNewValRef();
        
        void SetOldRowOffRef(std::vector<int> & old_arg);
        void SetOldColRef(std::vector<int> & old_arg);
        void SetOldValRef(std::vector<int> & old_arg);

        std::string toString();
        // These are for the next CSR
        std::vector<int> new_column_indices, new_row_offsets;
        // These are the current CSR
        std::vector<int> & old_column_indices_ref, & old_row_offsets_ref;
        int vertexCount;

};

#endif