#ifndef CSR_H
#define CSR_H

#include "SparseMatrix.h"
#include "COO.h"
#include "../VectorUtilities.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


class CSR : public SparseMatrix {
    public:
        CSR();
        CSR(COO & c);
        // Build first Graph from CSR
        CSR(CSR & c);
        // Copy constructor for allocating graph object
        CSR(const CSR & c);


        std::vector<int> * GetOldRowOffRef();
        std::vector<int> & GetNewRowOffRef();
        std::vector<int> * GetOldColRef();
        std::vector<int> & GetNewColRef();
        std::vector<int> * GetOldValRef();
        std::vector<int> & GetNewValRef();
        
        void SetOldRowOffRef(std::vector<int> & old_arg);
        void SetOldColRef(std::vector<int> & old_arg);
        void SetOldValRef(std::vector<int> & old_arg);

        void PopulateNewVals(int edgesLeftToCover);
        void PopulateNewRefs(int edgesLeftToCover);

        std::string toString();
        // These are for the next CSR
        std::vector<int> new_column_indices, new_row_offsets;
        // These are the current CSR
        std::vector<int> * old_column_indices_ref, * old_row_offsets_ref;
        int vertexCount;

        thrust::host_vector<int> col_vec;
        thrust::host_vector<int> row_offset;

        thrust::host_vector<int> dimensions_vec;
        thrust::host_vector<int> ones;

        thrust::device_vector<int> col_vec_dev;
        thrust::device_vector<int> row_offset_dev;

};

#endif
