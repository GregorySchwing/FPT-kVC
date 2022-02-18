
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
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

typedef r123::Philox4x32 RNG;
//static const double RAND_INTERVAL_GPU = 1.0/static_cast<double>(ULONG_MAX);

const int threadsPerBlock = 32;

__device__ int randomGPU(unsigned int counter, ulong step, ulong seed);
__device__ unsigned int h(unsigned int v);

__host__ __device__ int32_t Pack(int16_t a, int16_t b);
__host__ __device__ int16_t UnpackA(int32_t x);
__host__ __device__ int16_t UnpackB(int32_t x);
class Graph;

union VertexPair {
    // 2 16 bit unsigned integers
    unsigned  short yz[2];
};

__host__ void CalculateNewRowOffsets( int numberOfRows,
                                    int * global_row_offsets_dev_ptr,
                                    int * global_degrees_dev_ptr);

void CopyGraphToDeviceAndInduce( Graph & g,
                        int numberOfEdgesPerGraph,
                        int * global_row_offsets_dev_ptr,
                        int * global_columns_dev_ptr,
                        int * global_values_dev_ptr,
                        int * global_degrees_dev_ptr);

void CopyGraphFromDevice(Graph & g,
                        int numberOfEdgesPerGraph,
                        int * global_row_offsets_dev_ptr,
                        int * global_columns_dev_ptr,
                        int * host_row_offsets,
                        int * host_columns);

__global__ void launch_gpu_bfs_kernel( int N, int curr, int *levels,
                            int *nodes, int *edges, 
                            int * remaining,
                            int * colors,
                            int * color_finished,
                            int * predecessors,
                            int * finished);

__global__ void launch_gpu_bfs_color_kernel( int N, int curr, int *levels,
                                            int * colors,
                                            int * color_finished,
                                            int * predecessors);

__global__ void launch_gpu_bfs_finish_colors_kernel( int N, int curr, int *levels,
                                            int * colors,
                                            int * global_color_cardinalities,
                                            int * color_finished,
                                            int * predecessors);

__global__ void launch_gpu_color_finishing_kernel_1( int N,
                                                int * nodes,
                                                int * edges,
                                                int * colors,
                                                int * color_card,
                                                int * color_finished,
                                                int * middle_vertex);

__global__ void launch_gpu_color_finishing_kernel_2( int N,
                                                int * nodes,
                                                int * edges,
                                                int * colors,
                                                int * color_card,
                                                int * color_finished,
                                                int * middle_vertex);

__global__ void launch_gpu_combine_colors_kernel( int N,
                                                int k,
                                                int iter,
                                                int internal_iter,
                                                int * nodes,
                                                int * edges,
                                                int * M,
                                                int * U,
                                                int * U_Pred,
                                                int * colors,
                                                int * color_card);

__global__ void launch_gpu_sssp_coloring_1(int N,
                                        int k,
                                        int iter,
                                        int * M,
                                        int * U,
                                        int * U_Pred,
                                        int * colors,
                                        int * color_finished,
                                        int * middle_vertex);

__global__ void launch_gpu_sssp_coloring_maximize(int N,
                                        int k,
                                        int iter,
                                        int * M,
                                        int * U,
                                        int * U_Pred,
                                        int * colors,
                                        int * color_card,
                                        int * color_finished,
                                        int * middle_vertex);

__global__ void launch_gpu_sssp_coloring_2(int N,
                                        int k,
                                        int iter,
                                        int * M,
                                        int * U,
                                        int * U_Pred,
                                        int * colors,
                                        int * color_card,
                                        int * color_finished,
                                        int * middle_vertex);

void PerformSSSP(int numberOfRows,
                int root,
                int * global_row_offsets_dev_ptr,
                int * global_columns_dev_ptr,
                int * global_W,
                int * global_M,
                int * global_C,
                int * global_U,
                int * global_Pred,
                int * global_U_Pred);

// Currently root isn't assigned to any color.
void PerformPathPartitioning(int numberOfRows,
                            int k,
                            int root,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_middle_vertex,
                            int * global_M,
                            int * global_U_Prev,
                            int * global_U,
                            int * global_U_Pred,
                            int * global_colors,
                            int * global_color_card,
                            int * global_color_finished,
                            int * global_finished_card_reduced,
                            int * finished_gpu, 
                            int * nextroot_gpu);


__global__ void launch_gpu_sssp_kernel_1(   int N,             
                                            int * global_row_offsets_dev_ptr,
                                            int * global_columns_dev_ptr,
                                            int * global_W,
                                            int * global_M,
                                            int * global_C,
                                            int * global_U,
                                            int * U_Pred);


__global__ void launch_gpu_sssp_kernel_2(   int N,
                                            int * global_M,
                                            int * global_C,
                                            int * global_U,
                                            int * Pred,
                                            int * U_Pred);

__global__ void reset_partial_paths(int N,
                                    int * colors,
                                    int * color_card,
                                    int * color_finished,
                                    int * middle_vertex);

__global__ void calculate_percent_partitioned(int N,
                                                int * color_card,
                                                int * color_finished,
                                                int * finished_card_reduced);

void PerformBFS(int numberOfRows,
                int * global_levels,
                int * global_row_offsets_dev_ptr,
                int * global_columns_dev_ptr,
                int * triangle_counter_dev,
                int * global_colors_dev_ptr,
                int * global_color_cardinalities,
                int * global_color_finished_dev_ptr,
                int * global_predecessors);



void GetSource(int numberOfRows,
    int * triangle_counter_dev,
    int * source);

void PerformBFSColoring(int numberOfRows,
                int k,
                int * global_levels,
                int * global_row_offsets_dev_ptr,
                int * global_columns_dev_ptr,
                int * global_colors,
                int * global_color_card);

void SSSPAndBuildDepthCSR(Graph & gPrime, 
                        int root, 
                        int * host_levels,
                        int * new_row_offs_host,
                        int * new_cols_host,
                        int * new_row_offs,
                        int * new_cols,
                        int * new_colors,
                        int * host_U,
                        int * new_Pred,
                        int * new_color_finished);

__global__ void InduceSubgraph( int numberOfRows,
                                int edgesLeftToCover,
                                int * old_row_offsets_dev,
                                int * old_columns_dev,
                                int * old_values_dev,
                                int * global_row_offsets_dev_ptr,
                                int * global_columns_dev_ptr,
                                int * new_values_dev);

void CallInduceSubgraph(Graph & g, 
                    int * new_row_offs_dev,
                    int * new_cols_dev,
                    int * new_vals_dev,
                    int * new_degrees_dev,
                    int * new_row_offs_host,
                    int * new_cols_host,
                    int * new_vals_host);

void CallMIS(   Graph & g,
                int * new_row_offs_dev,
                int * new_cols_dev,
                int * new_vals_dev,
                int * new_degrees_dev,
                int * vertex_remaining);

void CallCountTriangles(
                        int numberOfRows,
                        int numberOfEdgesPerGraph,
                        int * numberOfTriangles_host,
                        int * new_row_offs_dev,
                        int * new_cols_dev,
                        int * new_row_offs_host,
                        int * new_cols_host,
                        int * triangle_counter_host,
                        int * triangle_counter_dev,
                        int * triangle_row_offsets_array_host,
                        int * triangle_row_offsets_array_dev);

void CallSaveTriangles( int numberOfRows,
                        int numberOfTriangles,
                        int * new_row_offs_dev,
                        int * new_cols_dev,
                        int * triangle_row_offsets_array_host,
                        int * triangle_row_offsets_array_dev,
                        int * triangle_candidates_a_host,
                        int * triangle_candidates_b_host,
                        int * triangle_candidates_a_dev,
                        int * triangle_candidates_b_dev);

void CallDisjointSetTriangles(
                        int numberOfRows,
                        int * new_row_offs_dev,
                        int * new_cols_dev,
                        int * triangle_row_offsets_array_dev,
                        int * triangle_counter_host,
                        int * triangle_counter_dev,
                        int * triangle_candidates_a_dev,
                        int * triangle_candidates_b_dev);

void CallColorTriangles(
                        int numberOfRows,
                        int * global_colors_dev_ptr,
                        int * global_color_finished_dev_ptr,
                        int * triangle_row_offsets_array_dev,
                        int * triangle_counter_dev,
                        int * triangle_candidates_a_dev,
                        int * triangle_candidates_b_dev);

__global__ void CountTriangleKernel(int numberOfRows,
                                    int * new_row_offs_dev,
                                    int * new_cols_dev,
                                    int * triangle_counter_dev);

__global__ void SaveTrianglesKernel(int numberOfRows,
                                    int * new_row_offs_dev,
                                    int * new_cols_dev,
                                    int * triangle_row_offsets_array_dev,
                                    int * triangle_candidates_a_dev,
                                    int * triangle_candidates_b_dev);

__global__ void CalculateConflictDegree(int numberOfRows,
                                        int * triangle_row_offsets_array_dev,
                                        int * triangle_counter_dev,
                                        int * triangle_candidates_a_dev,
                                        int * triangle_candidates_b_dev);

__global__ void CheckForConficts(int numberOfRows,
                                int * triangle_counter_dev,
                                int * conflictsRemain);

__global__ void CalculateConflictDegreeNeighborhoodSum(int numberOfRows,
                                        int * new_row_offs_dev,
                                        int * new_cols_dev,
                                        int * triangle_counter_dev,
                                        int * conflictDegreeNeighborhoodSum_dev);

__global__ void IdentifyMaximumConflictTriangles(int numberOfRows,
                                                int * triangle_row_offsets_array_dev,
                                                int * triangle_candidates_a_dev,
                                                int * triangle_candidates_b_dev,
                                                int * triangle_counter_dev,
                                                int * L_dev);

__global__ void TurnOffMaximumEdgeOfConflictTriangles(int numberOfRows,
                                                int * triangle_row_offsets_array_dev,
                                                int * triangle_candidates_a_dev,
                                                int * triangle_candidates_b_dev,                                                
                                                int * triangle_counter_dev,
                                                int * L_dev,
                                                int * M_dev);


__global__ void RemoveMaximumEdgeOfConflictTriangles(int numberOfRows,
                                                    int * triangle_row_offsets_array_dev,
                                                    int * triangle_candidates_a_dev,
                                                    int * triangle_candidates_b_dev,
                                                    int * M_dev);

__global__ void DisjointSetTriangleKernel(int numberOfRows,
                                    int * new_row_offs_dev,
                                    int * new_cols_dev,
                                    int * triangle_counter_dev,
                                    int * triangle_reduction_array_dev);

__global__ void ColorTriangleKernel(int numberOfRows,
                                    int * triangle_row_offsets_array_dev,
                                    int * triangle_candidates_a_dev,
                                    int * triangle_candidates_b_dev,
                                    int * triangle_counter_dev,
                                    int * colors,
                                    int * color_finished);

void Sum(int expanded_size,
        int * expanded,
        int * reduced);

void MaximizePathLength(int numberOfRows,
                        int k,
                        int iter,
                        int * finished_gpu,
                        int * global_M,
                        int * global_U,
                        int * global_U_Pred,
                        int * global_colors,
                        int * global_color_card,
                        int * global_color_finished,
                        int * global_middle_vertex);

void FindMaximumDistanceNonFinishedColor(int numberOfRows,
                                        int * global_colors,
                                        int * global_M,
                                        int * global_color_finished,
                                        int * global_U_Prev,
                                        int * global_U,
                                        int * nextroot_gpu);


__global__ void multiply_distance_by_finished_boolean(int N,
                                                    int * M,
                                                    int * colors,
                                                    int * color_finished,
                                                    int * U_Prev,
                                                    int * U);

void GetMaxDist(int N,
                int * M,
                int * nextroot_gpu);

#endif
#endif
