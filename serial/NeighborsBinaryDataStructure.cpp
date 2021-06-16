#include "NeighborsBinaryDataStructure.h"

NeighborsBinaryDataStructure::NeighborsBinaryDataStructure(CSR * compressedSparseMatrix){
    twoDimensionalBitMatrixOfNeighboringVertices.resize(compressedSparseMatrix->numberOfRows);
    for (int i = 0; i < compressedSparseMatrix->numberOfRows; ++i)
        twoDimensionalBitMatrixOfNeighboringVertices[i].resize(compressedSparseMatrix->numberOfRows);

    /* Should I neighbor myself? */
    for (int i = 0; i < compressedSparseMatrix->numberOfRows; ++i)
        for (int j = compressedSparseMatrix->row_offsets[i]; j < compressedSparseMatrix->row_offsets[i+1]; ++j){
            twoDimensionalBitMatrixOfNeighboringVertices[i][compressedSparseMatrix->column_indices[j]] = 1;
            twoDimensionalBitMatrixOfNeighboringVertices[compressedSparseMatrix->column_indices[j]][i] = 1;
        }

    for (int i = 0; i < compressedSparseMatrix->numberOfRows; ++i){
        std::string buffer;
        to_string(twoDimensionalBitMatrixOfNeighboringVertices[i], buffer);
        std::cout << buffer << std::endl;
    }
}

int NeighborsBinaryDataStructure::GetDegree(int vertex){
    return twoDimensionalBitMatrixOfNeighboringVertices[vertex].count();
}

void NeighborsBinaryDataStructure::InducedSubgraphAdjMatrixAsym(CSR * compressedSparseMatrix){
    /* Should I neighbor myself? */
    for (int i = 0; i < compressedSparseMatrix->numberOfRows; ++i)
        for (int j = compressedSparseMatrix->row_offsets[i]; j < compressedSparseMatrix->row_offsets[i+1]; ++j){
            twoDimensionalBitMatrixOfNeighboringVertices[i][compressedSparseMatrix->column_indices[j]] = compressedSparseMatrix->values[j];
            twoDimensionalBitMatrixOfNeighboringVertices[compressedSparseMatrix->column_indices[j]][i] = compressedSparseMatrix->values[j];
        }

    for (int i = 0; i < compressedSparseMatrix->numberOfRows; ++i){
        std::string buffer;
        to_string(twoDimensionalBitMatrixOfNeighboringVertices[i], buffer);
        std::cout << buffer << std::endl;
    }
}

void NeighborsBinaryDataStructure::InducedSubgraphAdjMatrixSym(CSR * compressedSparseMatrix){
    /* Should I neighbor myself? */
    for (int i = 0; i < compressedSparseMatrix->numberOfRows; ++i)
        for (int j = compressedSparseMatrix->row_offsets[i]; j < compressedSparseMatrix->row_offsets[i+1]; ++j){
            twoDimensionalBitMatrixOfNeighboringVertices[i][compressedSparseMatrix->column_indices[j]] = compressedSparseMatrix->values[j];
        }

    for (int i = 0; i < compressedSparseMatrix->numberOfRows; ++i){
        std::string buffer;
        to_string(twoDimensionalBitMatrixOfNeighboringVertices[i], buffer);
        std::cout << buffer << std::endl;
    }
}