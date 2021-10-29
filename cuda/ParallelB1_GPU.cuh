
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
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include "CUDAUtils.cuh"
#include "Graph.h"
#include "Random123/philox.h"
#include <stdint.h>

typedef r123::Philox4x32 RNG;
//static const double RAND_INTERVAL_GPU = 1.0/static_cast<double>(ULONG_MAX);

const int threadsPerBlock = 32;

__device__ int randomGPU(unsigned int counter, ulong step, ulong seed);

class Graph;

void CopyGraphToDevice( Graph & g,
                        int * new_row_offsets_dev_ptr,
                        int * new_columns_dev_ptr,
                        int * values_dev_ptr,
                        int * new_degrees_dev_ptr,
                        int numberOfEdgesPerGraph,
                        int * global_edges_left_to_cover_count,
                        int * global_remaining_vertices_dev_ptr,
                        int * global_remaining_vertices_size_dev_ptr,
                        int verticesRemainingInGraph);

void CallPopulateTree(int numberOfLevels, 
                    Graph & gPrime);

CUDA_HOSTDEV int CalculateWorstCaseSpaceComplexity(int vertexCount);
CUDA_HOSTDEV long long CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels);
CUDA_HOSTDEV long long CalculateSizeRequirement(int startingLevel,
                                                            int endingLevel);
CUDA_HOSTDEV long long CalculateLevelOffset(int level);
CUDA_HOSTDEV long long CalculateLevelUpperBound(int level);
CUDA_HOSTDEV long long CalculateDeepestLevelWidth(int maxLevel);

__global__ void  PrintEdges(int levelOffset,
                                    int levelUpperBound,
                                    int numberOfRows,
                                    int numberOfEdgesPerGraph,
                                    int * global_columns_tree,
                                    int * global_offsets_buffer,
                                    int * printAlt,
                                    int * printCurr);

__global__ void  PrintVerts(int levelOffset,
                                    int levelUpperBound,
                                    int numberOfRows,
                                    int * global_verts_tree,
                                    int * global_vertex_buffer,
                                    int * printAlt,
                                    int * printCurr);

__global__ void First_Graph_GPU(int vertexCount, 
                                int size,
                                int numberOfRows,
                                int * old_row_offsets_dev,
                                int * old_columns_dev,
                                int * old_values_dev,
                                int * new_row_offsets_dev,
                                int * new_columns_dev,
                                int * new_values_dev,
                                int * old_degrees_dev,
                                int * new_degrees_dev,
                                int * global_row_offsets_dev_ptr,
                                int * global_columns_dev_ptr,
                                int * global_values_dev_ptr,
                                int * global_degrees_dev_ptr
                                );

__device__ void AssignPointers(long long globalIndex,
                                long long edgesPerNode,
                                long long numberOfVertices,
                                int * new_row_offsets_dev,
                                int * new_columns_dev,
                                int * values_dev,
                                int * new_degrees_dev);

__global__ void PopulateTreeParallelLevelWise_GPU(int numberOfLevels, 
                                            long long edgesPerNode,
                                            long long numberOfVertices,
                                            int * new_row_offsets_dev,
                                            int * new_columns_dev,
                                            int * values_dev,
                                            int * new_degrees_dev);

__global__ void SetEdges(int numberOfRows,
                        int numberOfEdgesPerGraph,
                        int levelOffset,
                        int levelUpperBound,
                        int * global_row_offsets_dev_ptr,
                        int * global_columns_dev_ptr,
                        int * global_values_dev_ptr,
                        int * global_paths_ptr,
                        int * global_paths_length,
                        int * global_edges_left_to_cover_count);

__global__ void SetDegreesAndCountEdgesLeftToCover(int numberOfRows,
                                            int numberOfEdgesPerGraph,
                                            int levelOffset,
                                            int levelUpperBound,
                                            int * global_row_offsets_dev_ptr,
                                            int * global_values_dev_ptr,
                                            int * global_degrees_dev_ptr,
                                            int * global_edges_left_to_cover_count);

__global__ void InduceSubgraph( int numberOfRows,
                                int edgesLeftToCover,
                                int * old_row_offsets_dev,
                                int * old_columns_dev,
                                int * old_values_dev,
                                int * global_row_offsets_dev_ptr,
                                int * global_columns_dev_ptr,
                                int * new_values_dev);

__global__ void InduceRowOfSubgraphs( int numberOfRows,
                                      int levelOffset,
                                      int levelUpperBound,
                                      int numberOfEdgesPerGraph,
                                      int * global_edges_left_to_cover_count,                                      int * global_row_offsets_dev_ptr,
                                      int * global_columns_dev_ptr,
                                      int * global_values_dev_ptr
                                    );

__global__ void CalculateNewRowOffsets( int numberOfRows,
                                        int levelOffset,
                                        int levelUpperBound,
                                        int * global_row_offsets_dev_ptr,
                                        int * global_degrees_dev_ptr);

__global__ void CreateSubsetOfRemainingVerticesLevelWise(int levelOffset,
                                                int levelUpperBound,
                                                int numberOfRows,
                                                int * global_degrees_dev_ptr,
                                                int * global_vertices_remaining,
                                                int * global_vertices_remaining_count);

__global__ void DFSLevelWise(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int maxDegree,
                            int numberOfEdgesPerGraph,
                            int * global_degrees_dev_ptr,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_vertices_remaining,
                            int * global_vertices_remaining_count,
                            int * global_paths_ptr,
                            int * global_paths_length,
                            int * global_outgoing_edge_vertices,
                            int * global_outgoing_edge_vertices_count);

__global__ void GetRandomVertex(int levelOffset,
                                int levelUpperBound,
                                int numberOfRows,
                                int * global_remaining_vertices_dev_ptr,
                                int * global_remaining_vertices_size_dev_ptr,
                                int * global_paths_ptr);

__global__ void GetRandomVertexSharedMem(int levelOffset,
                                int levelUpperBound,
                                int numberOfRows,
                                int * global_remaining_vertices_dev_ptr,
                                int * global_remaining_vertices_size_dev_ptr,
                                int * global_paths_ptr);

__global__ void ParallelDFSRandom(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_degrees_dev_ptr,
                            int * global_paths_ptr,
                            int * global_nonpendant_path_bool_dev_ptr,
                            int * global_nonpendant_path_reduced_bool_dev_ptr,
                            int * global_nonpendant_child_dev_ptr);

__global__ void ParallelProcessPendantEdges(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_degrees_dev_ptr,
                            int * global_nonpendant_path_bool_dev_ptr,
                            int * global_pendant_child_dev_ptr);

__global__ void ParallelIdentifyNonIntersectingNonPendantPaths(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_pendant_path_bool_dev_ptr,
                            int * global_pendant_child_dev_ptr);

__global__ void ParallelProcessDegreeZeroVertices(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_degrees_dev_ptr);

__global__ void ParallelCreateLevelAwareRowOffsets(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_offsets_buffer);

__global__ void SetVerticesRemaingSegements(int deepestLevelSize,
                                            int numberOfRows,
                                            int * global_vertex_segments);

__global__ void ParallelQuicksortWithDNF(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_degrees_dev_ptr);
                            /*
__global__ void SerialProcessPendantEdge(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_paths_ptr,
                            int * global_nonpendant_path_bool_dev_ptr);
*/

__device__ void SetOutgoingEdges(int rowOffsOffset,
                                int valsAndColsOffset,
                                int degreesOffset,
                                int u,
                                int * global_row_offsets_dev_ptr,
                                int * global_columns_dev_ptr,
                                int * global_values_dev_ptr,
                                int * global_degrees_dev_ptr);

__device__ void SetIncomingEdges(int rowOffsOffset,
                                int valsAndColsOffset,
                                int degreesOffset,
                                int u,
                                int * global_row_offsets_dev_ptr,
                                int * global_columns_dev_ptr,
                                int * global_values_dev_ptr,
                                int * global_degrees_dev_ptr);

__global__ void GenerateChildren(int leafIndex,
                                int numberOfRows,
                                int maxDegree,
                                int numberOfEdgesPerGraph,
                                int * global_row_offsets_dev_ptr,
                                int * global_columns_dev_ptr,
                                int * global_values_dev_ptr,
                                int * global_degrees_dev_ptr,
                                int * global_vertices_remaining,
                                int * global_paths_ptr,
                                int * global_vertices_remaining_count,
                                int * global_outgoing_edge_vertices);
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
