#ifndef COO_H
#define COO_H
#include "SparseMatrix.h"
#include <cstdlib> /* rand */
#include <vector>
#include <string>
#include <iostream>

class COO final : public SparseMatrix
{
public:
    COO(int size, int numberOfRows, int numberOfColumns, bool populate = false);
    std::string toString();
    void sortMyself();
    bool getIsSorted() const { return isSorted; }
    void insertElements(const SparseMatrix & c);
    COO& SpMV(COO & c);
    std::vector<int> column_indices, row_indices;
private:
bool isSorted;
};
#endif