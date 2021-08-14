
#ifndef Parallel_B1_GPU_H
#define Parallel_B1_GPU_H

#include <cuda.h>
#include "Graph_GPU.cuh"

class Graph_GPU;

__host__ __device__ int CalculateWorstCaseSpaceComplexity(int vertexCount);
__host__ __device__ long long CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels);
__host__ __device__ long long CalculateSizeRequirement(int startingLevel,
                                                            int endingLevel);
__host__ __device__ long long CalculateLevelOffset(int level);

__device__ void AssignPointers(long long globalIndex,
                                long long edgesPerNode,
                                long long numberOfVertices,
                                Graph_GPU ** graphs,
                                int ** new_row_offsets_dev,
                                int ** new_columns_dev,
                                int ** values_dev,
                                int ** new_degrees_dev);

__global__ void PopulateTreeParallelLevelWise(int numberOfLevels, 
                                            long long edgesPerNode,
                                            long long numberOfVertices,
                                            Graph_GPU ** graphs,
                                            int ** new_row_offsets_dev,
                                            int ** new_columns_dev,
                                            int ** values_dev,
                                            int ** new_degrees_dev);
/*
__host__ __device__ void PopulateTree(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
__host__ __device__ void PopulateTreeParallel(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
__host__ __device__ int PopulateTreeParallelLevelWise(int numberOfLevels, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
__host__ __device__ void PopulateTreeParallelAsymmetric(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer);
__host__ __device__ int GenerateChildren( Graph & child_g);

__host__ __device__ void TraverseUpTree(int index, 
                            std::vector<Graph> & graphs,
                            std::vector<int> & answer);
*/


/*
    __host__ __device__ void DFS(std::vector<int> & new_row_off,
                    std::vector<int> & new_col_ref,
                    std::vector<int> & new_vals_ref,
                    std::vector<int> & path, 
                    int rootVertex);
    __host__ __device__ int GetRandomVertex(std::vector<int> & verticesRemaining);
    __host__ __device__ int GetRandomOutgoingEdge(std::vector<int> & new_row_off,
                                    std::vector<int> & new_col_ref,
                                    std::vector<int> & new_values_ref,
                                    int v, 
                                    std::vector<int> & path);
    __host__ __device__ int classifyPath(std::vector<int> & path);
    __host__ __device__ void createVertexSetsForEachChild(std::vector< std::vector<int> > & childrensVertices, 
                                                int caseNumber, 
                                                std::vector<int> & path);
*/


#endif
