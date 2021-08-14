#include "CSR_GPU.cuh"

/* Copy constructor */
CSR_GPU::CSR_GPU(const CSR & c):
SparseMatrix_GPU(c)
{
    new_row_offsets.reserve(c.new_row_offsets.capacity());
    new_column_indices.reserve(c.new_column_indices.capacity());
}
