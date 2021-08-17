
#ifndef Parallel_B1_GPU_H
#define Parallel_B1_GPU_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#pragma once
#ifdef FPT_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include "Graph_GPU.cuh"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include "CUDAUtils.cuh"

class Graph_GPU;

void CopyGraphToDevice(Graph & gPrime, Graph_GPU * g_dev);

void CallPopulateTree(int numberOfLevels, 
                    Graph & gPrime);

CUDA_HOSTDEV int CalculateWorstCaseSpaceComplexity(int vertexCount);
CUDA_HOSTDEV long long CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels);
CUDA_HOSTDEV long long CalculateSizeRequirement(int startingLevel,
                                                            int endingLevel);
CUDA_HOSTDEV long long CalculateLevelOffset(int level);

__global__ void First_Graph_GPU(Graph_GPU * g_dev,
                                int vertexCount, 
                                int size,
                                int numberOfRows,
                                int ** old_row_offsets_dev,
                                int ** old_columns_dev,
                                int ** old_values_dev,
                                int ** new_row_offsets_dev,
                                int ** new_columns_dev,
                                int ** new_values_dev,
                                int ** old_degrees_dev,
                                int ** new_degrees_dev);

__global__ void CopyBackGraph(Graph_GPU * g_dev, int * internal_dev_ptr, int * sizedev2host);


__device__ void AssignPointers(long long globalIndex,
                                long long edgesPerNode,
                                long long numberOfVertices,
                                Graph_GPU ** graphs,
                                int ** new_row_offsets_dev,
                                int ** new_columns_dev,
                                int ** values_dev,
                                int ** new_degrees_dev);

__global__ void PopulateTreeParallelLevelWise_GPU(Graph_GPU * gPrime,
                                            int numberOfLevels, 
                                            long long edgesPerNode,
                                            long long numberOfVertices,
                                            Graph_GPU ** graphs,
                                            int ** new_row_offsets_dev,
                                            int ** new_columns_dev,
                                            int ** values_dev,
                                            int ** new_degrees_dev);

__global__ void InitGPrime_GPU(Graph_GPU * g_dev, 
array_container * mpt, array_container * S, Graph_GPU * root);

// Fill a perfect 3-ary tree to a given depth
__global__ void TearDownTree_GPU(int numberOfLevels, 
                                Graph_GPU ** graphs);
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
#endif