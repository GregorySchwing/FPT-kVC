#include "CSR_GPU.cuh"

/* Copy constructor */
__host__ CSR_GPU::CSR_GPU(const CSR & c):
SparseMatrix_GPU(c)
{
    new_row_offsets_dev.reserve(c.new_row_offsets.capacity());
    new_column_indices_dev.reserve(c.new_column_indices.capacity());
}

__host__ CSR_GPU::~CSR_GPU()
{
    new_row_offsets_dev.clear();
    new_column_indices_dev.clear();
    new_row_offsets_dev.shrink_to_fit();
    new_column_indices_dev.shrink_to_fit();
}
