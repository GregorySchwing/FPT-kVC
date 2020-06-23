#ifndef DCSR_H
#define DCSR_H
#include <cstdlib> /* rand */
#include <vector>
#include "COO.h"
#include "CSR.h"
#include "VectorUtilities.h"

class DCSR : SparseMatrix
{
public:
    DCSR(int size, int sparsity_factor, int numberOfSegments, int sizeOfSegments, int numberOfRows, int numberOfColumns);
    DCSR(const CSR & c, int sizeOfSegments);
    void insertElements(SparseMatrix & s);

    void insertElements(    int row,
                            const std::vector<int> & B_offsets,
                            const std::vector<int> & B_cols, 
                            const std::vector<int> & B_vals,
                            const std::vector<int> & B_sizes    
                        );
    void allocateSegments(  const std::vector<int> & B_sizes,
                            const std::vector<int> & B_offsets, 
                            const std::vector<int> & B_cols, 
                            const std::vector<int> & B_vals);
    void allocateSegments(CSR & c);

    std::vector<int> getRowSizes(const CSR & c);
    std::vector<int> CSRRowOffsetsToDCSRFormat(const CSR & c);

    std::string toString();

private:
std::vector<int> column_indices, row_offsets, row_sizes;
int * memory_allocation_pointer;
int sizeOfSegments;
int pitch;
int alpha;



int getInitializedBack(std::vector<int> offsets);

friend class COO;
};
#endif