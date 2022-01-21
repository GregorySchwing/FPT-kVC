
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
                        int numberOfEdgesPerGraph,
                        int * global_edges_left_to_cover_count,
                        int * new_row_offsets_dev_ptr,
                        int * new_columns_dev_ptr,
                        int * values_dev_ptr,
                        int verticesRemainingInGraph,
                        int * global_remaining_vertices_size_dev_ptr,
                        int * global_degrees_offsets_ptr,
                        int * new_degrees_dev_ptr,
                        int * global_remaining_vertices_dev_ptr,
                        int * global_vertex_is_remaining_flag_dev_ptr,
                        int * global_active_leaf_indices,
                        int * global_active_leaf_indices_count);

void CallPopulateTree(int numberOfLevels, 
                    Graph & gPrime);

CUDA_HOSTDEV int CalculateWorstCaseSpaceComplexity(int vertexCount);
CUDA_HOSTDEV long long CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels);
CUDA_HOSTDEV long long CalculateSizeRequirement(int startingLevel,
                                                            int endingLevel);
CUDA_HOSTDEV long long CalculateLevelOffset(int level);
CUDA_HOSTDEV long long CalculateLevelUpperBound(int level);
CUDA_HOSTDEV long long CalculateDeepestLevelWidth(int maxLevel);
CUDA_HOSTDEV int CalculateNumberOfFullLevels(int leavesThatICanGenerate);

CUDA_HOSTDEV int ClosedFormLevelDepth(int leavesThatICanGenerate);

//__global__ int ClosedFormLevelDepth(int leavesThatICanProcess);
#ifndef NDEBUG

__global__ void  PrintEdges(int activeVerticesCount,
                                    int numberOfRows,
                                    int numberOfEdgesPerGraph,
                                    int * global_columns_tree,
                                    int * global_offsets_buffer,
                                    int * printAlt,
                                    int * printCurr);

__global__ void  PrintVerts(int activeVerticesCount,
                                    int numberOfRows,
                                    int * global_verts_tree,
                                    int * global_vertex_buffer,
                                    int * printAlt,
                                    int * printCurr);

__global__ void  PrintRowOffs(int activeVerticesCount,
                                    int numberOfRows,
                                    int * printAlt,
                                    int * printCurr);                                    

__global__ void  PrintSets(int activeVerticesCount,
                            int * curr_paths_indices,
                            int * alt_paths_indices,
                            int * curr_set_inclusion,
                            int * alt_set_inclusion,
                           int * global_set_path_offsets);

__global__ void  PrintData(int activeVerticesCount,
                            int numberOfRows,
                            int numberOfEdgesPerGraph, 
                            int * row_offs,
                            int * cols,
                            int * vals,
                            int * degrees,
                            int * verts_remain,
                            int * edges_left,
                            int * verts_remain_count);
#endif

__global__ void SetEdges(const int numberOfRows,
                        const int numberOfEdgesPerGraph,
                        const int * global_active_leaf_indices,
                        const int * global_active_leaf_parent_leaf_index,
                        const int * global_row_offsets_dev_ptr,
                        const int * global_columns_dev_ptr,
                        int * global_values_dev_ptr,
                        int * global_degrees_dev_ptr,
                        int * global_edges_left_to_cover_count,
                        int * verts_remain_count,
                        const int * global_vertices_included_dev_ptr);

__global__ void SetPendantEdges(const int numberOfRows,
                        const int numberOfEdgesPerGraph,
                        const int * global_row_offsets_dev_ptr,
                        const int * global_columns_dev_ptr,
                        int * global_values_dev_ptr,
                        int * global_degrees_dev_ptr,
                        int * global_edges_left_to_cover_count,
                        const int * global_pendant_path_bool_dev_ptr,
                        const int * global_pendant_child_dev_ptr,
                        int * verts_remain_count);

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

__global__ void ParallelDFSRandom(int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int verticesRemainingInGraph,
                            int * global_active_leaf_indices,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_degrees_offsets_dev_ptr,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_degrees_dev_ptr,
                            int * global_paths_ptr,
                            int * global_paths_indices_ptr,
                            int * global_nonpendant_path_bool_dev_ptr,
                            int * global_nonpendant_path_reduced_bool_dev_ptr,
                            int * global_nonpendant_child_dev_ptr,
                            int * global_edges_left_to_cover_count,
                            int * global_verts_remain_count);

__global__ void ParallelIdentifyVertexDisjointNonPendantPathsClean(
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_active_leaf_indices,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_pendant_path_bool_dev_ptr,
                            int * global_pendant_child_dev_ptr,
                            int * global_set_inclusion_bool_ptr,
                            int * global_reduced_set_inclusion_count_ptr,
                            int * edges_left,
                            int * verts_remain_count);

__global__ void ParallelProcessDegreeZeroVerticesClean(
                            int numberOfRows,
                            int verticesRemainingInGraph,
                            int * global_active_leaf_indices,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_edges_left_to_cover_count,
                            int * global_degrees_dev_ptr);

__global__ void ParallelRowOffsetsPrefixSumDevice(int numberOfEdgesPerGraph,
                                                int numberOfRows,
                                                int * global_row_offsets_dev_ptr,
                                                int * global_cols_vals_segments);


__global__ void SetVerticesRemaingSegements(int dLSPlus1,
                                            int numberOfRows,
                                            int * global_vertex_segments);
*/
__global__ void SetPathOffsets(int sDLSPlus1,
                               int * global_set_path_offsets);

__global__ void ParallelAssignMISToNodesBreadthFirstClean(int * global_active_leaf_indices,
                                        int * global_set_paths_indices,
                                        int * global_reduced_set_inclusion_count_ptr,
                                        int * global_paths_ptr,
                                        int * global_vertices_included_dev_ptr,
                                        int * edges_left,
                                        int * verts_remain_count);

__global__ void ParallelCalculateOffsetsForNewlyActivateLeafNodesBreadthFirst(
                                        int * global_active_leaves_count_current,
                                        int * global_active_leaves_count_new,
                                        int * global_reduced_set_inclusion_count_ptr,
                                        int * global_reduced_set_newly_active_leaves_count_ptr,
                                        int * edges_left,
                                        int * verts_remain_count);

__global__ void ParallelPopulateNewlyActivateLeafNodesBreadthFirstClean(
                                        int * global_active_leaves,
                                        int * global_newly_active_leaves,
                                        int * global_active_leaves_count_current,
                                        int * global_reduced_set_inclusion_count_ptr,
                                        int * global_newly_active_offset_ptr,
                                        int * global_active_leaf_index,
                                        int * global_active_leaf_parent_leaf_index,
                                        int * global_active_leaf_parent_leaf_value,
                                        int * edges_left,
                                        int * verts_remain_count);

#endif
#endif
