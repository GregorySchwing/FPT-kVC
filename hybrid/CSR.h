#ifndef CSR_H
#define CSR_H

#include "SparseMatrix.h"
#include "COO.h"
#include "../VectorUtilities.h"
#include <thrust/host_vector.h>


class CSR : public SparseMatrix {
    public:
        CSR();
        CSR(COO & c);
        // Build first Graph from CSR
        CSR(CSR & c);
        // Copy constructor for allocating graph object
        CSR(const CSR & c);


        thrust::host_vector<int> * GetOldRowOffRef();
        thrust::host_vector<int> & GetNewRowOffRef();
        thrust::host_vector<int> * GetOldColRef();
        thrust::host_vector<int> & GetNewColRef();
        thrust::host_vector<int> * GetOldValRef();
        thrust::host_vector<int> & GetNewValRef();
        
        void SetOldRowOffRef(thrust::host_vector<int> & old_arg);
        void SetOldColRef(thrust::host_vector<int> & old_arg);
        void SetOldValRef(thrust::host_vector<int> & old_arg);

        void PopulateNewVals(int edgesLeftToCover);
        void PopulateNewRefs(int edgesLeftToCover);

        std::string toString();
        // These are for the next CSR
        thrust::host_vector<int> new_column_indices, new_row_offsets;
        // These are the current CSR
        thrust::host_vector<int> * old_column_indices_ref, * old_row_offsets_ref;
        int vertexCount;

};

#endif