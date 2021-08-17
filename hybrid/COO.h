#ifndef COO_H
#define COO_H
#include "SparseMatrix.h"
#include <cstdlib> /* rand */
#include <vector>
#include <string>
#include <iostream>
#include "../lib/CSVRange.h"
#include  <iterator>
#include <algorithm> /* rand */
#include <map>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

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
    void addEdge(int u, int v, int weight);
    void addEdgeSimple(int u, int v, int weight);
    void addEdgeASymmetric(int u, int v, int weight);
    void addEdgeSymmetric(int u, int v, int weight);
    void addEdgeToMap(int u, int v, int weight);
    int GetNumberOfVertices();
    int GetNumberOfEdges();
    void BuildTheExampleCOO();
    void BuildCycleCOO();
    void BuildCOOFromFile(std::string filename);
    void SetVertexCountFromEdges();
    //COO& SpMV(COO & c);
    thrust::host_vector<int> new_column_indices, new_row_indices;
    int vertexCount;

private:
std::map< std::pair<int, int>, int > orderedMap;
bool isSorted;
};
#endif
