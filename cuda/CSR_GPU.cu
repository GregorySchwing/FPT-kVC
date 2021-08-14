#include "CSR_GPU.cuh"

/* Copy constructor */
__host__ CSR_GPU::CSR_GPU(const CSR & c):
SparseMatrix_GPU(c)
{
    new_row_offsets_dev.reserve(c.new_row_offsets.capacity());
    new_column_indices_dev.reserve(c.new_column_indices.capacity());
}
