
#ifndef Parallel_B1_GPU_H
#define Parallel_B1_GPU_H

#include "Graph_GPU.cuh"
#include <iostream>


class ParallelB1_GPU {
public:
__host__ __device__ int static CalculateWorstCaseSpaceComplexity(int vertexCount);
__host__ __device__ long long static CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels);
__host__ __device__ long long static CalculateSizeRequirement(int startingLevel,
                                                            int endingLevel);
/*
__host__ __device__ void static PopulateTree(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
__host__ __device__ void static PopulateTreeParallel(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
__host__ __device__ int static PopulateTreeParallelLevelWise(int numberOfLevels, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
__host__ __device__ void static PopulateTreeParallelAsymmetric(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
__host__ __device__ int static GenerateChildren( Graph & child_g);

__host__ __device__ void static TraverseUpTree(int index, 
                            std::vector<Graph> & graphs,
                            std::vector<int> & answer);
*/


private:
/*
    __host__ __device__ void static DFS(std::vector<int> & new_row_off,
                    std::vector<int> & new_col_ref,
                    std::vector<int> & new_vals_ref,
                    std::vector<int> & path, 
                    int rootVertex);
    __host__ __device__ int static GetRandomVertex(std::vector<int> & verticesRemaining);
    __host__ __device__ int static GetRandomOutgoingEdge(std::vector<int> & new_row_off,
                                    std::vector<int> & new_col_ref,
                                    std::vector<int> & new_values_ref,
                                    int v, 
                                    std::vector<int> & path);
    __host__ __device__ int static classifyPath(std::vector<int> & path);
    __host__ __device__ void static createVertexSetsForEachChild(std::vector< std::vector<int> > & childrensVertices, 
                                                int caseNumber, 
                                                std::vector<int> & path);
*/
};

#endif
