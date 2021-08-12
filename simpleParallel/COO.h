#ifndef COO_H
#define COO_H
#include "SparseMatrix.h"
#include <cstdlib> /* rand */
#include <vector>
#include <string>
#include <iostream>
#include "../common/CSVRange.h"
#include  <iterator>
#include <algorithm> /* rand */
#include <unordered_map>

// A hash function used to hash a pair of any kind
struct hash_pair {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const
    {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ hash2;
    }
};

class COO final : public SparseMatrix
{
public:
    COO(int size, int numberOfRows, int numberOfColumns, bool populate = false);
    COO(int numberOfRows, int numberOfColumns);
    /* Simple Copy Constructor */
    COO(COO & coo_arg);
    /* SpRef Operation */
    COO(COO & coo_arg, int edgesLeftToCover);
    /* From file */
    COO();

    std::string toString();
    void sortMyself();
    bool getIsSorted() const { return isSorted; }
    void insertElements(const SparseMatrix & c);
    void addEdge(int u, int v, int weight, int edgeID);
    void addEdgeSimple(int u, int v, int weight);
    void addEdgeASymmetric(int u, int v, int weight);
    void addEdgeSymmetric(int u, int v, int weight);
    void addEdgeToMap(int u, int v, int weight);
    int GetNumberOfVertices();
    int GetNumberOfEdges();
    void BuildTheExampleCOO();
    void BuildCOOFromFile(std::string filename);
    void SetVertexCountFromEdges();
    //COO& SpMV(COO & c);
    std::vector<int> new_column_indices, new_row_indices;
    int vertexCount;

private:
std::unordered_map<std::pair<int, int>, bool, hash_pair> um;
bool isSorted;
};
#endif