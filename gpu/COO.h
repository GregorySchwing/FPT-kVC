#ifndef COO_H
#define COO_H
#include "SparseMatrix.h"
#include <cstdlib> /* rand */
#include <vector>
#include <string>
#include <iostream>
#include <cuda.h>

class COO final : public SparseMatrix
{
public:
    COO(int size, int numberOfRows, int numberOfColumns, bool populate = false);
    std::string toString();
    void sortMyself();
    bool getIsSorted() const { return isSorted; }
    void insertElements(const SparseMatrix & c);
    COO& SpMV(COO & c);
    int * column_indices, * row_indices;
private:
void wrapperFunction(int * column_indices, int * row_indices, int * values, int size);
bool isSorted;
};
#endif