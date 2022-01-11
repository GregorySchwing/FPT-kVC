#ifdef FPT_CUDA

#include "ParallelB1_GPU.cuh"
#include <math.h>       /* pow */
#include "cub/cub.cuh"
#include "Random123/boxmuller.hpp"
// For viz
#include "../lib/DotWriter/lib/DotWriter.h"
#include <map>
// Sum of i = 0 to n/2
// 3^i

__device__ int randomGPU(unsigned int counter, ulong step, ulong seed)
{
  RNG::ctr_type c = {{}};
  RNG::ukey_type uk = {{}};
  uk[0] = step;
  uk[1] = seed;
  RNG::key_type k = uk;
  c[0] = counter;
  RNG::ctr_type r = philox4x32(c, k);
  return r[0];
}

__device__ inline double randomGPUDouble(unsigned int counter, ulong step, ulong seed)
{
  RNG::ctr_type c = {{}};
  RNG::ukey_type uk = {{}};
  uk[0] = step;
  uk[1] = seed;
  RNG::key_type k = uk;
  c[0] = counter;
  RNG::ctr_type r = philox4x32(c, k);
  return r123::u01<double>(r[0]);
}

__device__ RNG::ctr_type randomGPU_four(unsigned int counter, ulong step, ulong seed)
{
  RNG::ctr_type c = {{}};
  RNG::ukey_type uk = {{}};
  uk[0] = step;
  uk[1] = seed;
  RNG::key_type k = uk;
  c[0] = counter;
  RNG::ctr_type r = philox4x32(c, k);
  return r;
}

__host__ __device__ int CalculateWorstCaseSpaceComplexity(int vertexCount){
    int summand= 0;
    // ceiling(vertexCount/2) loops
    for (int i = 0; i < (vertexCount + 2 - 1)/2; ++i)
        summand += pow (3.0, i);
    return summand;
}

__host__ __device__ long long CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels){
    long long summand= 0;
    // ceiling(vertexCount/2) loops
    for (int i = 0; i < NumberOfLevels; ++i)
        summand += pow (3.0, i);
    return summand;
}

__host__ __device__ long long CalculateDeepestLevelWidth(int deepestLevelSize){
    long long summand= 0;
    summand += pow (3.0, deepestLevelSize);
    return summand;
}

__host__ __device__ int CalculateNumberOfFullLevels(int leavesThatICanProcess){
    //
    // Level 0 : Size 1
    // Level 1 : Size 3
    // Level 2 : Size 9
    // Level 3 : Size 27
    // Level 4 : Size 81
    // Level 5 : Size 243
    // Level 6 : Size 729
    // Level 7 : Size 2187
    if (leavesThatICanProcess / 1 == 0)
        return 0;
    else if (leavesThatICanProcess / 4 == 0)
        return 1;
    else if (leavesThatICanProcess / 13 == 0)
        return 2;
    else if (leavesThatICanProcess / 40 == 0)
        return 3;
    else if (leavesThatICanProcess / 121 == 0)
        return 4;
    else if (leavesThatICanProcess / 364 == 0)
        return 5;
    else if (leavesThatICanProcess / 1093 == 0)
        return 6;
    else if (leavesThatICanProcess / 3280 == 0)
        return 7;
    else
        return -1;
    // Current max number of threads per block is 2048
    // Each thread can find a path, therefore, 
    // there shouldnt be a case where a vertex can generate
    // greater than 2048*3 = 6144 leaves 
    // The next number in the succession would be
    // 9841 leaves.

}


// This calculates the width of the deepest level possibly generated
// based off the amount of threads per block.
__host__ int CalculateAmountOfMemoryToAllocatePerActiveLeaf(){
    int maximumAmountOfNewActiveLeavesPerLeaf = pow(ceil(log(threadsPerBlock)/log(3.0)), 3.0);
    return maximumAmountOfNewActiveLeavesPerLeaf;
}



__host__ __device__ int CalculateNumberInIncompleteLevel(int leavesThatICanProcess){
    //
    // Level 0 : Size 1
    // Level 1 : Size 3
    // Level 2 : Size 9
    // Level 3 : Size 27
    // Level 4 : Size 81
    // Level 5 : Size 243
    // Level 6 : Size 729
    // Level 7 : Size 2187
    if (leavesThatICanProcess / 1 == 0)
        return 0;
    else if (leavesThatICanProcess / 4 == 0)
        return leavesThatICanProcess - 1;
    else if (leavesThatICanProcess / 13 == 0)
        return leavesThatICanProcess - 4;
    else if (leavesThatICanProcess / 40 == 0)
        return leavesThatICanProcess - 13;
    else if (leavesThatICanProcess / 121 == 0)
        return leavesThatICanProcess -40;
    else if (leavesThatICanProcess / 364 == 0)
        return leavesThatICanProcess - 121;
    else if (leavesThatICanProcess / 1093 == 0)
        return leavesThatICanProcess - 364;
    else if (leavesThatICanProcess / 3280 == 0)
        return leavesThatICanProcess - 1093;
    else
        return -1;
    // Current max number of threads per block is 2048
    // Each thread can find a path, therefore, 
    // there shouldnt be a case where a vertex can generate
    // greater than 2048*3 = 6144 leaves 
    // The next number in the succession would be
    // 9841 leaves.

}

__host__ __device__ int ClosedFormLevelDepthIncomplete(int leavesThatICanProcess){
    //2*leavesThatICanProcess + 1 < 3^(n+1)
    // log(2*leavesThatICanProcess + 1) / log(3) < n + 1
    // log(2*leavesThatICanProcess + 1) / log(3) - 1 < n
    //return ceil(logf(2*leavesThatICanProcess + 1) / logf(3) - 1);
    return ceil(logf(2*leavesThatICanProcess + 1) / logf(3));
}

__host__ __device__ int ClosedFormLevelDepthComplete(int leavesThatICanProcess){
    //2*leavesThatICanProcess + 1 < 3^(n+1)
    // log(2*leavesThatICanProcess + 1) / log(3) < n + 1
    // log(2*leavesThatICanProcess + 1) / log(3) - 1 < n
    //return floor(logf(2*leavesThatICanProcess + 1) / logf(3) - 1);
    return floor(logf(2*leavesThatICanProcess + 1) / logf(3));
}

__host__ __device__ int TreeSize(int levels){
    return (1.0 - pow(3.0, levels+1.0))/(1.0 - 3.0);
}

/*
__global__ void ClosedFormLevelDepth(int leavesThatICanProcess){
    //2*leavesThatICanProcess + 1 < 3^(n+1)
    // log(2*leavesThatICanProcess + 1) / log(3) < n + 1
    // log(2*leavesThatICanProcess + 1) / log(3) - 1 < n
    return ceil(logf(2*leavesThatICanProcess + 1) / logf(3) - 1);
}
*/
__global__ void  PrintEdges(int activeVerticesCount,
                                    int numberOfRows,
                                    int numberOfEdgesPerGraph,
                                    int * global_columns_tree,
                                    int * global_column_buffer,
                                    int * printAlt,
                                    int * printCurr){
    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;
    printf("Tree\n");
    for (int i = 0; i < (activeVerticesCount)*numberOfEdgesPerGraph; ++i){
        printf("%d ",global_columns_tree[i]);
    }
    printf("\n");

    printf("Buffer\n");
    for (int i = 0; i < (activeVerticesCount)*numberOfEdgesPerGraph; ++i){
        printf("%d ",global_column_buffer[i]);
    }
    printf("\n");

    printf("Unsorted\n");

    for (int i = 0; i < (activeVerticesCount)*numberOfEdgesPerGraph; ++i){
        printf("%d ",printAlt[i]);
    }
    printf("\n");

    printf("Sorted\n");
    for (int i = 0; i < (activeVerticesCount)*numberOfEdgesPerGraph; ++i){
        printf("%d ",printCurr[i]);
    }
    printf("\n");
}

__global__ void  PrintVerts(int activeVerticesCount,
                                    int numberOfRows,
                                    int * printAlt,
                                    int * printCurr){
    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;
    printf("Tree\n");
    printf("Unsorted\n");

    for (int g = 0; g < (activeVerticesCount); ++g){
        printf("\n");
        for (int i = 0; i < numberOfRows; ++i){
            printf("%d ",printAlt[g*numberOfRows + i]);
        }
    }
    printf("\n");

    printf("Sorted\n");
    for (int g = 0; g < (activeVerticesCount); ++g){
        printf("\n");
        for (int i = 0; i < numberOfRows; ++i){
            printf("%d ",printCurr[g*numberOfRows + i]);
        }
    }
    printf("\n");
}

__global__ void  PrintRowOffs(int activeVerticesCount,
                                    int numberOfRows,
                                    int * printAlt,
                                    int * printCurr){
    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;
    printf("Tree\n");
    printf("Unsorted\n");

    for (int g = 0; g < (activeVerticesCount); ++g){
        printf("\n");
        for (int i = 0; i < numberOfRows+1; ++i){
            printf("%d ",printAlt[g*(numberOfRows+1) + i]);
        }
    }
    printf("\n");

    printf("Sorted\n");
    for (int g = 0; g < (activeVerticesCount); ++g){
        printf("\n");
        for (int i = 0; i < numberOfRows+1; ++i){
            printf("%d ",printCurr[g*(numberOfRows+1) + i]);
        }
    }
    printf("\n");
}

__global__ void  PrintSets(int activeVerticesCount,
                            int * curr_paths_indices,
                            int * alt_paths_indices,
                            int * curr_set_inclusion,
                            int * alt_set_inclusion,
                           int * global_set_path_offsets){
    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;
    printf("Tree\n");
    for (int v = 0; v < activeVerticesCount; ++v){
        printf("Index : %d; Offsets (%d, %d)\n", v, global_set_path_offsets[v], global_set_path_offsets[v+1]);
        for (int i = 0; i < threadsPerBlock; ++i){
            printf("%d ",curr_paths_indices[v*threadsPerBlock + i]);
        }
        printf("\n");
        for (int i = 0; i < threadsPerBlock; ++i){
            printf("%d ",curr_set_inclusion[v*threadsPerBlock + i]);
        }
        printf("\n");
    }

    for (int v = 0; v < activeVerticesCount; ++v){
        printf("Index : %d\n", v);
        for (int i = 0; i < threadsPerBlock; ++i){
            printf("%d ",alt_paths_indices[v*threadsPerBlock + i]);
        }
        printf("\n");
        for (int i = 0; i < threadsPerBlock; ++i){
            printf("%d ",alt_set_inclusion[v*threadsPerBlock + i]);
        }
        printf("\n");
    }
    
}

__global__ void  PrintData(int activeVerticesCount,
                            int numberOfRows,
                            int numberOfEdgesPerGraph, 
                            int verticesRemainingInGraph,
                            int * row_offs,
                            int * cols,
                            int * vals,
                            int * degrees,
                            int * verts_remain,
                            int * global_edges_left_to_cover_count,
                            int * verts_remain_count){
    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;
    for (int g = 0; g < (activeVerticesCount); ++g){
        printf("Graph %d\n", g);
        printf("Row Offs\n");
        for (int i = 0; i < numberOfRows+1; ++i){
            printf("%d ",row_offs[g*(numberOfRows+1) + i]);
        }
        printf("\n");
        printf("Cols\n");
        for (int i = 0; i < numberOfEdgesPerGraph; ++i){
            printf("%d ",cols[g*numberOfEdgesPerGraph + i]);
        }
        printf("\n");
        printf("Vals\n");
        for (int i = 0; i < numberOfEdgesPerGraph; ++i){
            printf("%d ",vals[g*numberOfEdgesPerGraph + i]);
        }
        printf("\n");
        printf("Degrees\n");
        for (int i = 0; i < numberOfRows; ++i){
            printf("%d ",degrees[g*(numberOfRows) + i]);
        }
        printf("\n");
        printf("Verts Rem\n");
        for (int i = 0; i < verticesRemainingInGraph; ++i){
            printf("%d ",verts_remain[g*(verticesRemainingInGraph) + i]);
        }
        printf("\n");
        printf("Edges left: %d\n", global_edges_left_to_cover_count[g]);
        printf("\n");
        printf("Vertices Remaining Count: %d\n", verts_remain_count[g]);
    }
}

__host__ __device__ long long CalculateSizeRequirement(int startingLevel,
                                                        int endingLevel){
    long long summand= 0;
    // ceiling(vertexCount/2) loops
    for (int i = startingLevel; i < endingLevel; ++i)
        summand += pow (3.0, i);
    return summand;
}

__host__ __device__ long long CalculateLevelOffset(int level){
    if(level == 0)
        return 0;
    else
        return pow(3.0, (level-1)) + 1;
}

__host__ __device__ long long CalculateLevelUpperBound(int level){
    if(level == 0)
        return 1;
    else
        return pow(3.0, (level)) + 1;
}

__host__ __device__ long long CalculateLevelSize(int level){
    if(level == 0)
        return 0;
    else 
        return pow(3.0, level);
}

__host__ void SortPathIndices(int activeVerticesCount,
                              int threadsPerBlock,
                              cub::DoubleBuffer<int> & paths_indices,
                              cub::DoubleBuffer<int> & set_inclusion,
                              int * global_set_path_offsets){
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    int num_items = activeVerticesCount*threadsPerBlock;
    int num_segments = activeVerticesCount;

    // Since vertices in a level follow each other, we reuse gob iterator
    // When we have active vertices from different levels, we will need 2 iterators
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, set_inclusion, paths_indices, 
        num_items, num_segments, global_set_path_offsets, global_set_path_offsets + 1);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, set_inclusion, paths_indices, 
        num_items, num_segments, global_set_path_offsets, global_set_path_offsets + 1);

    cudaFree(d_temp_storage);

}

__host__ void CUBLibraryPrefixSumDevice(int * activeVerticesCount,
                                        cub::DoubleBuffer<int> & active_leaf_offset){
    // Declare, allocate, and initialize device-accessible pointers for input and output
    int  num_items = activeVerticesCount[0]+1;      // e.g., 7
    int  *d_in = active_leaf_offset.Current();        // e.g., [8, 6, 7, 5, 3, 0, 9]
    int  *d_out = active_leaf_offset.Alternate();         // e.g., [ ,  ,  ,  ,  ,  ,  ]
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // d_out s<-- [0, 8, 14, 21, 26, 29, 29]
    cudaFree(d_temp_storage);

}

__global__ void ParallelRowOffsetsPrefixSumDevice(int numberOfEdgesPerGraph,
                                                int numberOfRows,
                                                int * global_row_offsets_dev_ptr,
                                                int * global_cols_vals_segments){

    int leafIndex = blockIdx.x; 

    //printf("LevelAware RowOffs blockIdx %d is running\n", blockIdx.x);
    //printf("LevelAware RowOffs leaf index %d is running\n", leafIndex);

    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int bufferRowOffsOffset = leafIndex * (numberOfRows + 1);

    for (int iter = threadIdx.x; iter < numberOfRows+1; iter += blockDim.x){
        global_cols_vals_segments[bufferRowOffsOffset + iter] = (leafIndex * numberOfEdgesPerGraph) + global_row_offsets_dev_ptr[rowOffsOffset + iter];
        //printf("global_cols_vals_segments[bufferRowOffsOffset + %d] = %d + %d\n", iter, (leafIndex * numberOfEdgesPerGraph), global_row_offsets_dev_ptr[rowOffsOffset + iter]);
    }
}

__host__ void RestoreDataStructuresAfterRemovingChildrenVertices(int activeVerticesCount,
                                                                int threadsPerBlock,
                                                                int numberOfRows,
                                                                int numberOfEdgesPerGraph,
                                                                int verticesRemainingInGraph,
                                                                cub::DoubleBuffer<int> & row_offsets,
                                                                cub::DoubleBuffer<int> & columns,
                                                                cub::DoubleBuffer<int> & values,
                                                                cub::DoubleBuffer<int> & remaining_vertices,
                                                                int * global_cols_vals_segments,
                                                                int * global_vertex_segments){

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    int num_items = (activeVerticesCount)*numberOfEdgesPerGraph;
    int num_segments = (activeVerticesCount)*(numberOfRows+1);

    // Since vertices in a level follow each other, we reuse gob iterator
    // When we have active vertices from different levels, we will need 2 iterators
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, values, columns,
        num_items, num_segments, global_cols_vals_segments, global_cols_vals_segments + 1);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, values, columns,
        num_items, num_segments, global_cols_vals_segments, global_cols_vals_segments + 1);

    cudaFree(d_temp_storage);

    int * printAlt = values.Alternate();
    int * printCurr = values.Current();
/*
    PrintEdges<<<1,1>>>  (activeVerticesCount,
                numberOfRows,
                numberOfEdgesPerGraph,
                columns.Alternate(),
                columns.Current(),
                printAlt,
                printCurr);
*/
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
                // Determine temporary device storage requirements
    void     *d_temp_storage2 = NULL;
    temp_storage_bytes = 0;
    num_items = (activeVerticesCount)*verticesRemainingInGraph;
    num_segments = activeVerticesCount;

    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage2, temp_storage_bytes, remaining_vertices,
        num_items, num_segments, global_vertex_segments, global_vertex_segments + 1);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage2, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage2, temp_storage_bytes, remaining_vertices,
        num_items, num_segments, global_vertex_segments, global_vertex_segments + 1);

    cudaFree(d_temp_storage2);
/*
    printAlt = remaining_vertices.Alternate();
    printCurr = remaining_vertices.Current();

    PrintVerts<<<1,1>>> (activeVerticesCount,
                numberOfRows,
                printAlt,
                printCurr);
*/
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

}

__host__ void GetMaxLeafValue(int activeVerticesCount,
                              cub::DoubleBuffer<int> & active_leaves_value,
                              int * max){

    // Declare, allocate, and initialize device-accessible pointers for input and output
    int  num_items = activeVerticesCount;      // e.g., 7
    int  *d_in = active_leaves_value.Current();          // e.g., [8, 6, 7, 5, 3, 0, 9]
    int * maxTmp;
    cudaMalloc(&maxTmp, 1 * sizeof(int));
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, maxTmp, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run max-reduction
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, maxTmp, num_items);
    // d_out <-- [9]
    cudaFree(d_temp_storage);

    cudaMemcpy(max, maxTmp, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
}

__host__ void GetMinLeafValue(int activeVerticesCount,
                                cub::DoubleBuffer<int> & active_leaves_value,
                                int * min){

    // Declare, allocate, and initialize device-accessible pointers for input and output
    int  num_items = activeVerticesCount;      // e.g., 7
    int  *d_in = active_leaves_value.Current();          // e.g., [8, 6, 7, 5, 3, 0, 9]
    int * minTmp;
    cudaMalloc(&minTmp, 1 * sizeof(int));

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, minTmp, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run max-reduction
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, minTmp, num_items);
    // d_out <-- [9]
    cudaFree(d_temp_storage);

    cudaMemcpy(min, minTmp, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

}

typedef int inner_array_t[2];

__global__ void InduceSubgraph( int numberOfRows,
                                int edgesLeftToCover,
                                int * old_row_offsets_dev,
                                int * old_columns_dev,
                                int * old_values_dev,
                                int * new_row_offsets_dev,
                                int * new_columns_dev,
                                int * new_values_dev){

    //int row = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.x;

    inner_array_t *C_ref = new inner_array_t[numberOfRows];

    for (int iter = row; iter < numberOfRows; iter += blockDim.x){

        //printf("Thread %d, row %d", threadIdx.x, iter);
        C_ref[iter][0] = 0;
        C_ref[iter][1] = 0;
        //printf("Thread %d, row %d, old_row_offsets_dev[iter] = %d", threadIdx.x, iter, old_row_offsets_dev[iter]);
        //printf("Thread %d, row %d, old_row_offsets_dev[iter+1] = %d", threadIdx.x, iter, old_row_offsets_dev[iter+1]);
        //printf("Thread %d, row %d, old_values_dev[endOffset] = %d", threadIdx.x, iter, old_values_dev[old_row_offsets_dev[iter+1]]);

        int beginIndex = old_row_offsets_dev[iter];
        int endIndex = old_row_offsets_dev[iter+1];

        for (int i = beginIndex; i < endIndex; ++i){
            ++C_ref[iter][old_values_dev[i]];
        }

        // This is  [old degree - new degree , new degree]
        for (int i = 1; i < 2; ++i){
            C_ref[iter][i] = C_ref[iter][i] + C_ref[iter][i-1];
        }
        //printf("Thread %d, row %d, almost done", threadIdx.x, iter);

        /* C_ref[A_row_indices[i]]]-1 , because the values of C_ref are from [1, n] -> [0,n) */
        for (int i = endIndex-1; i >= beginIndex; --i){
            if (old_values_dev[i]){
                new_columns_dev[new_row_offsets_dev[iter] - C_ref[iter][0] + C_ref[iter][1]-1] = old_columns_dev[i];
                new_values_dev[new_row_offsets_dev[iter] - C_ref[iter][0] + C_ref[iter][1]-1] = old_values_dev[i];
                --C_ref[iter][old_values_dev[i]];
            }
        }
        if (row == 0){
            printf("Block %d induced root of graph", blockIdx.x);
            for (int i = 0; i < edgesLeftToCover; ++i){
                printf("%d ",new_columns_dev[i]);
            }
            printf("\n");
            for (int i = 0; i < edgesLeftToCover; ++i){
                printf("%d ",new_values_dev[i]);
            }
            printf("\n");
        }
    }
    delete[] C_ref;
}

__global__ void SetEdges(const int numberOfRows,
                        const int numberOfEdgesPerGraph,
                        const int * global_active_leaf_value,
                        const int * global_active_leaf_parent_leaf_index,
                        const int * global_row_offsets_dev_ptr,
                        const int * global_columns_dev_ptr,
                        int * global_values_dev_ptr,
                        int * global_degrees_dev_ptr,
                        int * global_edges_left_to_cover_count,
                        int * global_verts_remain_count,
                        const int * global_vertices_included_dev_ptr){

    int leafIndex = blockIdx.x;
    if (0 == global_edges_left_to_cover_count[leafIndex] || 0 == global_verts_remain_count[leafIndex])
        return;
    int leafValue = global_active_leaf_value[leafIndex];
    int originalLV = leafValue;
    int parentLeafValue = global_active_leaf_parent_leaf_index[leafIndex];
    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int valsAndColsOffset = leafIndex * numberOfEdgesPerGraph;
    int degreesOffset = leafIndex * numberOfRows;
    int searchTreeIndex = leafValue*2;
    int LB, UB, v, vLB, vUB, myChild;
    // Number of loops could explicitly calculated based on parent and leaf val
    while(leafValue != parentLeafValue){
        for (int childIndex = 0; childIndex < 2; ++childIndex){
            searchTreeIndex = leafValue*2;
            myChild = global_vertices_included_dev_ptr[searchTreeIndex + childIndex];
            // Set out-edges
            LB = global_row_offsets_dev_ptr[rowOffsOffset + myChild];
            UB = global_row_offsets_dev_ptr[rowOffsOffset + myChild + 1]; 
            for (int edge = LB + threadIdx.x; edge < UB; edge += blockDim.x){
                // Since there are only 2 edges b/w each node,
                // We can safely decrement the target node's degree
                // If these are atomic, then duplicate myChildren isn't a problem
                // Since we'd be decrementing by 0 the second, third, ...etc time 
                // a duplicate myChild was processed.
                //global_degrees_dev_ptr[degreesOffset + 
                //    global_columns_dev_ptr[valsAndColsOffset + edge]] 
                //        -= global_values_dev_ptr[valsAndColsOffset + edge];
                atomicAdd(&global_degrees_dev_ptr[degreesOffset + 
                    global_columns_dev_ptr[valsAndColsOffset + edge]],
                    -1*global_values_dev_ptr[valsAndColsOffset + edge]);
                    if (global_degrees_dev_ptr[degreesOffset + 
                        global_columns_dev_ptr[valsAndColsOffset + edge]] < 0){
                            printf("BAD WENT BELOW 0%d", myChild);
                        }
                // This avoids a reduction of the degrees array to get total edges
                atomicAdd(&global_edges_left_to_cover_count[leafIndex], -2*global_values_dev_ptr[valsAndColsOffset + edge]);
                global_values_dev_ptr[valsAndColsOffset + edge] = 0;
            }
            __syncthreads();
            if (threadIdx.x == 0){
                global_degrees_dev_ptr[degreesOffset + myChild] = 0;
            }
            __syncthreads();
            if (threadIdx.x == 0){
                printf("Block %d, leafValue %d, originalLV %d, parentLeafValue %d,  myChild removed %d\n", blockIdx.x, leafValue, originalLV, parentLeafValue, myChild);
            }
            // (u,v) is the form of edge pairs.  We are traversing over v's outgoing edges, 
            // looking for u as the destination and turning off that edge.
            // this may be more elegantly handled by 
            // (1) an associative data structure
            // (2) an undirected graph 
            // Parallel implementations of both of these need to be investigated.
            bool foundChild, tmp;
            LB = global_row_offsets_dev_ptr[rowOffsOffset + myChild];
            UB = global_row_offsets_dev_ptr[rowOffsOffset + myChild + 1];    // Set out-edges
            // There are two possibilities for parallelization here:
            // 1) Each thread will take an out edge, and then each thread will scan the edges leaving 
            // that vertex for the original vertex.
            //for (int edge = LB + threadIdx.x; edge < UB; edge += blockDim.x){

            // Basically, each thread is reading wildly different data
            // 2) 1 out edge is traversed at a time, and then all the threads scan
            // all the edges leaving that vertex for the original vertex.
            // This is the more favorable data access pattern.
            for (int edge = LB; edge < UB; ++edge){
                v = global_columns_dev_ptr[valsAndColsOffset + edge];
                // guarunteed to only have one incoming and one outgoing edge connecting (x,y)
                // All outgoing edges were set and are separated from this method by a __syncthreads
                // Thus there is no chance of decrementing the degree of the same node simulataneously
                vLB = global_row_offsets_dev_ptr[rowOffsOffset + v];
                vUB = global_row_offsets_dev_ptr[rowOffsOffset + v + 1];
                for (int outgoingEdgeOfV = vLB + threadIdx.x; 
                        outgoingEdgeOfV < vUB; 
                            outgoingEdgeOfV += blockDim.x){

                        foundChild = myChild == global_columns_dev_ptr[valsAndColsOffset + outgoingEdgeOfV];
                        // Set in-edge
                        // store edge status
                        tmp = global_values_dev_ptr[valsAndColsOffset + outgoingEdgeOfV];
                        //   foundChild     tmp   (foundChild & tmp)  (foundChild & tmp)^tmp
                        //1)      0          0            0                       0
                        //2)      1          0            0                       0
                        //3)      0          1            0                       1
                        //4)      1          1            1                       0
                        //
                        // Case 1: isnt myChild and edge is off, stay off
                        // Case 2: is myChild and edge is off, stay off
                        // Case 3: isn't myChild and edge is on, stay on
                        // Case 4: is myChild and edge is on, turn off
                        // All this logic is necessary because we aren't using degree to set upperbound
                        // we are using row offsets, which may include some edges turned off on a previous
                        // pendant edge processing step.
                        global_values_dev_ptr[valsAndColsOffset + outgoingEdgeOfV] ^= (foundChild & tmp);
                }
            }
            // Neccessary since just because the two children vertices form a bridge
            // doesnt mean they dont also share an edge
            __syncthreads();
        }
        leafValue = (leafValue-1)/3;
    }
}

__global__ void SetPendantEdges(const int numberOfRows,
                        const int numberOfEdgesPerGraph,
                        const int * global_row_offsets_dev_ptr,
                        const int * global_columns_dev_ptr,
                        int * global_values_dev_ptr,
                        int * global_degrees_dev_ptr,
                        int * global_edges_left_to_cover_count,
                        const int * global_pendant_path_bool_dev_ptr,
                        const int * global_pendant_child_dev_ptr,
                        int * global_verts_remain_count){

    int leafIndex = blockIdx.x;
    if (0 == global_edges_left_to_cover_count[leafIndex] || 0 == global_verts_remain_count[leafIndex])
        return;

    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int valsAndColsOffset = leafIndex * numberOfEdgesPerGraph;
    int degreesOffset = leafIndex * numberOfRows;
    // Only allocated for one level, not tree global
    int globalPendantPathBoolOffset = blockIdx.x * blockDim.x;
    int globalPendantPathChildOffset = globalPendantPathBoolOffset;
    int LB, UB, v, vLB, vUB, pendantChild, pendantBool;
    for (int row = 0; row < blockDim.x; ++row) {
        pendantChild = global_pendant_child_dev_ptr[globalPendantPathChildOffset + row];
        pendantBool = global_pendant_path_bool_dev_ptr[globalPendantPathChildOffset + row];
        if(!pendantBool)
            continue;
        __syncthreads();
        // Set out-edges
        LB = global_row_offsets_dev_ptr[rowOffsOffset + pendantChild];
        UB = global_row_offsets_dev_ptr[rowOffsOffset + pendantChild + 1]; 
        for (int edge = LB + threadIdx.x; edge < UB; edge += blockDim.x){
            // Since there are only 2 edges b/w each node,
            // We can safely decrement the target node's degree
            // If these are atomic, then duplicate pendantChild isn't a problem
            // Since we'd be decrementing by 0 the second, third, ...etc time 
            // a duplicate pendantChild was processed.
            atomicAdd(&global_degrees_dev_ptr[degreesOffset + 
                global_columns_dev_ptr[valsAndColsOffset + edge]],
                -1*global_values_dev_ptr[valsAndColsOffset + edge]);
            if (global_degrees_dev_ptr[degreesOffset + 
                global_columns_dev_ptr[valsAndColsOffset + edge]] < 0){
                    printf("BAD WENT BELOW 0%d", pendantChild);
                }
            // If I can guaruntee there is no duplicate pendant
            // the atomic can be removed
            //global_degrees_dev_ptr[degreesOffset + 
            //    global_columns_dev_ptr[valsAndColsOffset + edge]] 
            //        -= global_values_dev_ptr[valsAndColsOffset + edge];
            // This avoids a reduction of the degrees array to get total edges
            atomicAdd(&global_edges_left_to_cover_count[leafIndex], -2*global_values_dev_ptr[valsAndColsOffset + edge]);
            global_values_dev_ptr[valsAndColsOffset + edge] = 0;
        }
        __syncthreads();
        if (threadIdx.x == 0){
            global_degrees_dev_ptr[degreesOffset + pendantChild] = 0;
        }
        __syncthreads();
        // (u,v) is the form of edge pairs.  We are traversing over v's outgoing edges, 
        // looking for u as the destination and turning off that edge.
        // this may be more elegantly handled by 
        // (1) an associative data structure
        // (2) an undirected graph 
        // Parallel implementations of both of these need to be investigated.
        bool foundChild, tmp;
        LB = global_row_offsets_dev_ptr[rowOffsOffset + pendantChild];
        UB = global_row_offsets_dev_ptr[rowOffsOffset + pendantChild + 1];    // Set out-edges
        // There are two possibilities for parallelization here:
        // 1) Each thread will take an out edge, and then each thread will scan the edges leaving 
        // that vertex for the original vertex.
        //for (int edge = LB + threadIdx.x; edge < UB; edge += blockDim.x){

        // Basically, each thread is reading wildly different data
        // 2) 1 out edge is traversed at a time, and then all the threads scan
        // all the edges leaving that vertex for the original vertex.
        // This is the more favorable data access pattern.
        for (int edge = LB; edge < UB; ++edge){
            v = global_columns_dev_ptr[valsAndColsOffset + edge];
            // guarunteed to only have one incoming and one outgoing edge connecting (x,y)
            // All outgoing edges were set and are separated from this method by a __syncthreads
            // Thus there is no chance of decrementing the degree of the same node simulataneously
            vLB = global_row_offsets_dev_ptr[rowOffsOffset + v];
            vUB = global_row_offsets_dev_ptr[rowOffsOffset + v + 1];
            for (int outgoingEdgeOfV = vLB + threadIdx.x; 
                    outgoingEdgeOfV < vUB; 
                        outgoingEdgeOfV += blockDim.x){

                    foundChild = pendantChild == global_columns_dev_ptr[valsAndColsOffset + outgoingEdgeOfV];
                    // Set in-edge
                    // store edge status
                    tmp = global_values_dev_ptr[valsAndColsOffset + outgoingEdgeOfV];
                    //   foundChild     tmp   (foundChild & tmp)  (foundChild & tmp)^tmp
                    //1)      0          0            0                       0
                    //2)      1          0            0                       0
                    //3)      0          1            0                       1
                    //4)      1          1            1                       0
                    //
                    // Case 1: isnt pendantChild and edge is off, stay off
                    // Case 2: is pendantChild and edge is off, stay off
                    // Case 3: isn't pendantChild and edge is on, stay on
                    // Case 4: is pendantChild and edge is on, turn off
                    // All this logic is necessary because we aren't using degree to set upperbound
                    // we are using row offsets, which may include some edges turned off on a previous
                    // pendant edge processing step.
                    global_values_dev_ptr[valsAndColsOffset + outgoingEdgeOfV] ^= (foundChild & tmp);
            }
        }
        __syncthreads();
    }
}


// Sets the new degrees without the edges and the edges left to cover
__global__ void SetDegreesAndCountEdgesLeftToCover(int numberOfRows,
                            int numberOfEdgesPerGraph,
                            
                            int * global_row_offsets_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_degrees_dev_ptr,
                            int * global_edges_left_to_cover_count){

    int leafIndex = blockIdx.x;

    extern __shared__ int degrees[];

    // Use parent's row offs
    int rowOffsOffset = (numberOfRows + 1) * (leafIndex-1)/3;
    // my vals some of which are now 0
    int valsAndColsOffset = numberOfEdgesPerGraph * leafIndex;
    int degreesOffset = leafIndex * numberOfRows;
    int LB, UB, iter, row, edge;

    row = threadIdx.x;
    for (iter = row; iter < numberOfRows; iter += blockDim.x){
        LB = global_row_offsets_dev_ptr[rowOffsOffset + iter];
        UB = global_row_offsets_dev_ptr[rowOffsOffset + iter + 1];   
        for (edge = LB; edge < UB; ++edge)
            degrees[iter] += global_values_dev_ptr[valsAndColsOffset + edge];
        // Maybe this can be done asyncronously or after the whole array has been filled
        // I definitely need the sync threads since I modify the shared memory for memcpy
        // alternatively I could create two shared mem arrays, one for async write to global
        // and 1 for reduction
        global_degrees_dev_ptr[degreesOffset + iter] = degrees[iter];
    }
    __syncthreads();
    if (threadIdx.x == 0){
        for (int i = 0; i < numberOfRows; ++i)
            printf("leafIndex %d, blockID %d vertex %d, degree %d\n", leafIndex, blockIdx.x, i, degrees[i]);
        printf("\n");
    }
    int halvedArray = numberOfRows/2;
    while (halvedArray != 0) {
        // Neccessary since numberOfRows is likely greater than blockSize
        for (iter = row; iter < halvedArray; iter += blockDim.x){
            if (iter < halvedArray){
                degrees[iter] += degrees[iter + halvedArray];
            }
        }
        __syncthreads();
        halvedArray /= 2;
    }
    if (row == 0)
        global_edges_left_to_cover_count[leafIndex] = degrees[0];
}

__global__ void InduceRowOfSubgraphs( int numberOfRows,
                                      int levelOffset,
                                      int levelUpperBound,
                                      int numberOfEdgesPerGraph,
                                      int * global_edges_left_to_cover_count,
                                      int * global_row_offsets_dev_ptr,
                                      int * global_columns_dev_ptr,
                                      int * global_values_dev_ptr
                                    ){

    int leafIndex = levelOffset + blockIdx.x;
    if (leafIndex >= levelUpperBound) return;
    int rowOffsOffset;
    if (levelOffset == 0)
        rowOffsOffset = 0;
    else
        rowOffsOffset = (leafIndex-1)/3 * (numberOfRows + 1);
    int valsAndColsOffset = leafIndex * numberOfEdgesPerGraph;

    // Since three children share a parent, it is sensible for the old pointers to be shared memory
    // and for each block to induce three children
    // For now it still global..
    int * old_row_offsets_dev = &(global_row_offsets_dev_ptr[rowOffsOffset]);
    int * old_columns_dev = &(global_columns_dev_ptr[valsAndColsOffset]);
    int * old_values_dev = &(global_values_dev_ptr[valsAndColsOffset]);
    int * new_row_offsets_dev = &(global_row_offsets_dev_ptr[rowOffsOffset]);


    inner_array_t *C_ref = new inner_array_t[numberOfRows];
    for (int child = 1; child <= 3; ++child){
        int row = threadIdx.x;
        int newValsAndColsOffset = (3*leafIndex + child) * numberOfEdgesPerGraph;

        int * new_columns_dev =  &(global_columns_dev_ptr[newValsAndColsOffset]);
        int * new_values_dev = &(global_values_dev_ptr[newValsAndColsOffset]);
        for (int iter = row; iter < numberOfRows; iter += blockDim.x){

            //printf("Thread %d, row %d", threadIdx.x, iter);
            C_ref[iter][0] = 0;
            C_ref[iter][1] = 0;
            //printf("Thread %d, row %d, old_row_offsets_dev[iter] = %d", threadIdx.x, iter, old_row_offsets_dev[iter]);
            //printf("Thread %d, row %d, old_row_offsets_dev[iter+1] = %d", threadIdx.x, iter, old_row_offsets_dev[iter+1]);
            //printf("Thread %d, row %d, old_values_dev[endOffset] = %d", threadIdx.x, iter, old_values_dev[old_row_offsets_dev[iter+1]]);

            int beginIndex = old_row_offsets_dev[iter];
            int endIndex = old_row_offsets_dev[iter+1];

            for (int i = beginIndex; i < endIndex; ++i){
                ++C_ref[iter][old_values_dev[i]];
            }

            // This is  [old degree - new degree , new degree]
            for (int i = 1; i < 2; ++i){
                C_ref[iter][i] = C_ref[iter][i] + C_ref[iter][i-1];
            }
           // printf("Thread %d, row %d, almost done", threadIdx.x, iter);

            /* C_ref[A_row_indices[i]]]-1 , because the values of C_ref are from [1, n] -> [0,n) */
            for (int i = endIndex-1; i >= beginIndex; --i){
                if (old_values_dev[i]){
                    new_columns_dev[new_row_offsets_dev[iter] - C_ref[iter][0] + C_ref[iter][1]-1] = old_columns_dev[i];
                    new_values_dev[new_row_offsets_dev[iter] - C_ref[iter][0] + C_ref[iter][1]-1] = old_values_dev[i];
                    --C_ref[iter][old_values_dev[i]];
                }
            }
            //printf("Thread %d, row %d, finished", threadIdx.x, iter);
        }
        __syncthreads();
        if (threadIdx.x == 0){
            printf("Block %d, levelOffset %d, leafIndex %d, induced child %d\n", blockIdx.x, levelOffset, leafIndex, 3*leafIndex + child);
            for (int i = 0; i < global_edges_left_to_cover_count[leafIndex]; ++i){
                printf("%d ",new_columns_dev[i]);
            }
            printf("\n");
            for (int i = 0; i < global_edges_left_to_cover_count[leafIndex]; ++i){
                printf("%d ",new_values_dev[i]);
            }
            printf("\n");
        }
    }
    delete[] C_ref;
}

__global__ void CalculateNewRowOffsets( int numberOfRows,
                                        int levelOffset,
                                        int levelUpperBound,
                                        int * global_row_offsets_dev_ptr,
                                        int * global_degrees_dev_ptr){
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int leafIndex = levelOffset + threadID;
    if (leafIndex >= levelUpperBound) return;
    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int degreesOffset = leafIndex * numberOfRows;

    int i = 0;
    printf("leafIndex %d, degreesOffset = %d\n", leafIndex, degreesOffset);
    printf("leafIndex %d, rowOffsOffset = %d\n", leafIndex, rowOffsOffset);
    printf("leafIndex %d, new_row_offsets_dev[%d] = %d\n", leafIndex, i, global_row_offsets_dev_ptr[rowOffsOffset]);

    global_row_offsets_dev_ptr[rowOffsOffset] = i;
    for (i = 1; i <= numberOfRows; ++i)
    {
        global_row_offsets_dev_ptr[rowOffsOffset + i] = global_degrees_dev_ptr[degreesOffset + i - 1] + global_row_offsets_dev_ptr[rowOffsOffset + i - 1];
        printf("leafIndex %d, new_row_offsets_dev[%d] = %d\n", leafIndex, i, global_row_offsets_dev_ptr[rowOffsOffset + i]);
        printf("leafIndex %d, global_row_offsets_dev_ptr[rowOffsOffset + %d - 1] = %d\n", leafIndex, i, global_row_offsets_dev_ptr[rowOffsOffset + i - 1]);
        printf("leafIndex %d, global_degrees_dev_ptr[degreesOffset + %d - 1] = %d\n", leafIndex, i, global_degrees_dev_ptr[degreesOffset + i - 1]);
    }
}


// It is very important to only use the value in global_pendant_child_dev_ptr[]
// if global_pendant_path_bool_dev_ptr[] is true.  Otherwhise this path shouldnt
// be processed in the next step.

__global__ void ParallelDFSRandom(
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int verticesRemainingInGraph,
                            int * global_active_leaf_value,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_degrees_dev_ptr,
                            int * global_paths_ptr,
                            int * global_paths_indices_ptr,
                            int * global_pendant_path_bool_dev_ptr,
                            int * global_pendant_path_reduced_bool_dev_ptr,
                            int * global_pendant_child_dev_ptr,
                            int * global_edges_left_to_cover_count,
                            int * global_verts_remain_count){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Entered DFS\n");
        printf("\n");
    }
    int leafIndex = blockIdx.x;
    if (0 == global_edges_left_to_cover_count[leafIndex] || 0 == global_verts_remain_count[leafIndex])
        return;
    int leafValue = global_active_leaf_value[leafIndex];

    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Set leafIndex\n");
        printf("\n");
    }

    // Initialize path indices
    global_paths_indices_ptr[leafIndex*blockDim.x + threadIdx.x] = threadIdx.x;
    // Initialized to 0, so will always perform DFS on first call
    // Subsequently, only perform DFS on pendant edges, so nonpendant false
    //if (global_pendant_path_bool_dev_ptr[leafIndex + threadIdx.x])
    //    return;
    int globalPathOffset = leafIndex * 4 * blockDim.x;
    int sharedMemPathOffset = threadIdx.x * 4;
    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int valsAndColsOffset = leafIndex * numberOfEdgesPerGraph;
    int degreesOffset = leafIndex * numberOfRows;
    int remainingVerticesOffset = leafIndex * verticesRemainingInGraph;
    extern __shared__ int pathsAndPendantStatus[];
    int isInvalidPathBooleanArrayOffset = blockDim.x * 4;
    int iteration = 0;
    RNG::ctr_type r;
    unsigned int counter = 0;
    ulong seed = threadIdx.x;
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Setup offsets\n");
        printf("\n");
    }
    unsigned int remainingVerticesSize = global_remaining_vertices_size_dev_ptr[leafIndex];
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("remainingVerticesSize %d\n", remainingVerticesSize);
        printf("\n");
    }
    int outEdgesCount;
    r = randomGPU_four(counter, leafValue, seed);
    // Random starting point
    pathsAndPendantStatus[sharedMemPathOffset + iteration] = global_remaining_vertices_dev_ptr[remainingVerticesOffset + (r[iteration] % remainingVerticesSize)];
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("pathsAndPendantStatus %d\n", pathsAndPendantStatus[sharedMemPathOffset + iteration]);
        printf("\n");
    }
    ++iteration;

    // Set random out at depth 1
    int randomVertRowOff = global_row_offsets_dev_ptr[rowOffsOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]];
    // Using degrees allow us to ignore the edges which have been turned off
    outEdgesCount = global_degrees_dev_ptr[degreesOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]];
    int printVert = pathsAndPendantStatus[sharedMemPathOffset + iteration - 1];
    int badVal = r[iteration] % outEdgesCount;
    // Assumes the starting point isn't degree 0
    if (outEdgesCount <= 0){
        printf("failed to get nonzero deg %d\n", pathsAndPendantStatus[sharedMemPathOffset + iteration]);
        printf("randomVertRowOff %d\n",randomVertRowOff);
        printf("outEdgesCount %d\n",outEdgesCount);
        printf("printVert %d\n",printVert);
    }
    if (randomVertRowOff > 1000){
        printf("pointing to a removed vertex %d\n", pathsAndPendantStatus[sharedMemPathOffset + iteration]);
    }
    pathsAndPendantStatus[sharedMemPathOffset + iteration] =  global_columns_dev_ptr[valsAndColsOffset + randomVertRowOff + (r[iteration] % outEdgesCount)];
    ++iteration;
    //    printf("(r[iteration] mod outEdgesCount) %d\n", (r[iteration] % outEdgesCount));

    //if (threadIdx.x == 0){
    //    printf("Block %d, leafValue %d, got through first 2 iterations\n", blockIdx.x, leafValue);
    //}
    // Depth 2 and 3
    for (; iteration < 4; ++iteration){
        randomVertRowOff = global_row_offsets_dev_ptr[rowOffsOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]];
        // Using degrees allow us to ignore the edges which have been turned off
        outEdgesCount = global_degrees_dev_ptr[degreesOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]];
        //outEdgesCount = global_row_offsets_dev_ptr[rowOffsOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1] + 1]
        //                - randomVertRowOff;
        pathsAndPendantStatus[sharedMemPathOffset + iteration] =  global_columns_dev_ptr[valsAndColsOffset + randomVertRowOff + (r[iteration] % outEdgesCount)];
        // OutEdgesCount != 1 means there is another path that isn't a simple cycle
        if(pathsAndPendantStatus[sharedMemPathOffset + iteration] == 
            pathsAndPendantStatus[sharedMemPathOffset + iteration - 2]
                && outEdgesCount != 1){
            pathsAndPendantStatus[sharedMemPathOffset + iteration] =  global_columns_dev_ptr[valsAndColsOffset + randomVertRowOff + ((r[iteration] + 1) % outEdgesCount)];
        }
    }
    // Check 0,2 and 1,3 for pendantness in my thread's path
    pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x] = (pathsAndPendantStatus[sharedMemPathOffset + 0] == pathsAndPendantStatus[sharedMemPathOffset + 2]);
    pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x] |= (pathsAndPendantStatus[sharedMemPathOffset + 1] == pathsAndPendantStatus[sharedMemPathOffset + 3]);
    //if (threadIdx.x == 0 && blockIdx.x == 0){
    //    printf("Block %d, leafValue %d, got through last 2 iterations\n", blockIdx.x, leafValue);
    //    printf("\n");
    //}
    printf("leafValue %d Thread %d (path %d -> %d -> %d -> %d) is %s\n", leafValue,threadIdx.x, 
                                                            pathsAndPendantStatus[sharedMemPathOffset + 0],
                                                            pathsAndPendantStatus[sharedMemPathOffset + 1],
                                                            pathsAndPendantStatus[sharedMemPathOffset + 2],
                                                            pathsAndPendantStatus[sharedMemPathOffset + 3],
                                                            pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x] ? "pendant" : "nonpendant");

    // Each thread has a different - sharedMemPathOffset
    // Copy each thread's path to global memory
    // There may be a better way to do this to avoid threads skipping 4.
    // Perhaps :
    // for (start = threadIDx.x; start < tpb*4; start += blockDimx)
    // global_paths_ptr[globalPathOffset + start] = pathsAndPendantStatus[start]
    // STRIDING - NOT OPTIMAL 
    global_paths_ptr[globalPathOffset + sharedMemPathOffset + 0] = pathsAndPendantStatus[sharedMemPathOffset + 0];
    global_paths_ptr[globalPathOffset + sharedMemPathOffset + 1] = pathsAndPendantStatus[sharedMemPathOffset + 1];
    global_paths_ptr[globalPathOffset + sharedMemPathOffset + 2] = pathsAndPendantStatus[sharedMemPathOffset + 2];
    global_paths_ptr[globalPathOffset + sharedMemPathOffset + 3] = pathsAndPendantStatus[sharedMemPathOffset + 3];

    // Copy each path's pendantness boolean to global memory
    // Global is level global, not tree global
    global_pendant_path_bool_dev_ptr[blockIdx.x*blockDim.x + threadIdx.x] = pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x];
    // We give Case 3 priority over Case 2,
    // Since the serial algorithm short-circuits 
    // upon finding a pendant edge

    // We know either 
    // Case 3 - length 2
    // v, v1
    //path[0] == path[2], desired child is v
    // If path[0] == path[2] then path[0] != path[2] is false
    // cI = (path[0] != path[2])
    // Hence, cI == 0, since false casted to int is 0
    // Therefore, v == path[cI]
    int childIndex = (int)(pathsAndPendantStatus[sharedMemPathOffset + 0] != pathsAndPendantStatus[sharedMemPathOffset + 2]);

    // or
    // Case 2 - length 3
    // v, v1, v2
    // if path[0] != path[2] was true, then path[1] == path[3]
    // cI == 1, since true casted to int is 1
    // Desired child is v1
    // Therefore, v1 == path[cI]
    // Set child or set -1 if nonpendant.  This is important for mixture of paths that are pen/nonpen and share common vertex
    global_pendant_child_dev_ptr[blockIdx.x*blockDim.x + threadIdx.x] = (int)((pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x]))* pathsAndPendantStatus[sharedMemPathOffset + childIndex] - 1*(int)(!pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x]);

    __syncthreads();

    int i = blockDim.x/2;
    // Checks for any nonpendant edge path exists
    while (i != 0) {
        if (threadIdx.x < i){
            pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x] += pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    // Only 1 bool for 32 paths
    if (threadIdx.x == 0)
        global_pendant_path_reduced_bool_dev_ptr[blockIdx.x] = pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x];
}

// Each node assigned threadsPerBlock blocks,  
// Up to threadsPerBlock pendant children are processed
// 1 pendant child per block
// outgoing and incoming edges of the pendant child 
// are processed at thread level
// Block immediately returns if nonpendant child 
// or duplicate pendant child and not the largest
// indexed instance of that child
__global__ void ParallelProcessPendantEdges(
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_active_leaf_value,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_degrees_dev_ptr,
                            int * global_edges_left_to_cover_count,
                            int * global_pendant_path_bool_dev_ptr,
                            int * global_pendant_child_dev_ptr){
    // Beginning of group of TPB pendant children
    int myBlockOffset = (blockIdx.x / blockDim.x) * blockDim.x;
    int myBlockIndex = blockIdx.x % blockDim.x;
    int leafIndex = (blockIdx.x / blockDim.x);
    int leafValue = global_active_leaf_value[leafIndex];
    if (myBlockIndex == 0 && threadIdx.x == 0){
        printf("leaf Value %d Started ParallelProcessPendantEdges\n", leafValue);
    }
    // Only process pendant edges
    // 1 block per path, up to TPB paths per node
    if (!global_pendant_path_bool_dev_ptr[blockIdx.x])
        return;
    if (threadIdx.x == 0){
        printf("leafValue %d path %d is pendant\n", leafValue,myBlockIndex);
    }
    // My child won't be set unless this block represents a valid pendant path
    // Could be a shared var
    int myChild = global_pendant_child_dev_ptr[blockIdx.x];
    if (threadIdx.x == 0){
        printf("leafValue %d path %d's child is %d\n", leafValue, myBlockIndex, myChild);
    }
    extern __shared__ int childrenAndDuplicateStatus[];
    // Write all 32 pendant children to shared memory
    // This offset works because it isnt treewise global, just levelwise
    childrenAndDuplicateStatus[threadIdx.x] = global_pendant_child_dev_ptr[myBlockOffset + threadIdx.x];
    __syncthreads();
    //if (myBlockIndex == 0){
    printf("leaf value %d path %d's pendant[%d] is %d\n", leafValue, myBlockIndex, threadIdx.x, childrenAndDuplicateStatus[threadIdx.x]);
    // See if myChild is duplicated, 1 vs all comparison written to shared memory
    // Also, if it is duplicated, only process the largest index duplicate
    // If it isn't duplicated, process the child.

    // By the cardinality of rational numbers, 
    // Obviously the path should be pendant...currently failing because there are paths including 18 which aren't pendant
    childrenAndDuplicateStatus[blockDim.x + threadIdx.x] = ((childrenAndDuplicateStatus[threadIdx.x] == myChild) 
                                                            && myBlockIndex < threadIdx.x);
    __syncthreads();
    printf("leaf value %d Block index %d's childrenAndDuplicateStatus[%d] is %d\n", leafValue, myBlockIndex, threadIdx.x, childrenAndDuplicateStatus[blockDim.x + threadIdx.x]);
    int i = blockDim.x/2;
    // Checks for any duplicate children which have a smaller index than their other self
    // Only the smallest instance of a duplication will be false for both
    // 1) childrenAndDuplicateStatus[threadIdx.x] == myChild)
    // 2) myBlockIndex < threadIdx.x

    // We have a growing mask from the second condition like so
    // MASK 0: 0 0 ... 0
    // MASK 1: 1 0 ... 0
    // MASK 2: 1 1 ... 0
    //        .
    //        .
    //        .
    // MASK 3: 1 1 . 1 0
    // By the pigeonhole principle, 
    // there are only so many smallest index duplications.
    // If there is 1, then all 32 threads are equal,
    // Since we are or-redudcing, every block will 
    // return true except the first one.
    // If there are two or more, then at least one
    // of the masks (1-31) will set it false, and 
    // all the masks larger than it will mask the 
    // non-smallest duplicate to true.

    // Finally, the fact that mask 31 has a zero
    // in the last index is ok, because this child is
    // either the smallest duplicate and therefore 
    // should be false and added to the cover
    // or it is not the smallest, and it will be masked
    // by one of the smaller masks (0-30).

    while (i != 0) {
        if (threadIdx.x < i){
            childrenAndDuplicateStatus[blockDim.x + threadIdx.x] |= childrenAndDuplicateStatus[blockDim.x + threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    __syncthreads();
    if (childrenAndDuplicateStatus[blockDim.x])
        return;

    if (threadIdx.x == 0)
        printf("leafValue %d Block index %d made it past the return\n", leafValue, myBlockIndex);


    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int valsAndColsOffset = leafIndex * numberOfEdgesPerGraph;
    int degreesOffset = leafIndex * numberOfRows;
    int LB, UB, v, vLB, vUB;
    // Set out-edges
    LB = global_row_offsets_dev_ptr[rowOffsOffset + myChild];
    UB = global_row_offsets_dev_ptr[rowOffsOffset + myChild + 1]; 
    if (threadIdx.x == 0){
        printf("leafValue %d block index %d Set offsets in PPP\n", leafValue, myBlockIndex);
    }   
    for (int edge = LB + threadIdx.x; edge < UB; edge += blockDim.x){
        // Since there are only 2 edges b/w each node,
        // We can safely decrement the target node's degree
        // If these are atomic, then duplicate myChildren isn't a problem
        // Since we'd be decrementing by 0 the second, third, ...etc time 
        // a duplicate myChild was processed.
        global_degrees_dev_ptr[degreesOffset + 
            global_columns_dev_ptr[valsAndColsOffset + edge]] 
                -= global_values_dev_ptr[valsAndColsOffset + edge];
        // This avoids a reduction of the degrees array to get total edges
        atomicAdd(&global_edges_left_to_cover_count[leafIndex], -2*global_values_dev_ptr[valsAndColsOffset + edge]);
        global_values_dev_ptr[valsAndColsOffset + edge] = 0;
    }

    if (threadIdx.x == 0){
            global_degrees_dev_ptr[degreesOffset + myChild] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0){
        printf("leafValue %d Block index %d Finished out edges PPP\n", leafValue, myBlockIndex);
    }  
    if (threadIdx.x == 0){
        printf("leafValue %d Block index %d removed myChild %d\n", leafValue, myBlockIndex, myChild);
    }
    // (u,v) is the form of edge pairs.  We are traversing over v's outgoing edges, 
    // looking for u as the destination and turning off that edge.
    // this may be more elegantly handled by 
    // (1) an associative data structure
    // (2) an undirected graph 
    // Parallel implementations of both of these need to be investigated.
    bool foundChild, tmp;
    LB = global_row_offsets_dev_ptr[rowOffsOffset + myChild];
    UB = global_row_offsets_dev_ptr[rowOffsOffset + myChild + 1];    // Set out-edges
    // There are two possibilities for parallelization here:
    // 1) Each thread will take an out edge, and then each thread will scan the edges leaving 
    // that vertex for the original vertex.
    //for (int edge = LB + threadIdx.x; edge < UB; edge += blockDim.x){

    // Basically, each thread is reading wildly different data
    // 2) 1 out edge is traversed at a time, and then all the threads scan
    // all the edges leaving that vertex for the original vertex.
    // This is the more favorable data access pattern.
    for (int edge = LB; edge < UB; ++edge){
        v = global_columns_dev_ptr[valsAndColsOffset + edge];
        // guarunteed to only have one incoming and one outgoing edge connecting (x,y)
        // All outgoing edges were set and are separated from this method by a __syncthreads
        // Thus there is no chance of decrementing the degree of the same node simulataneously
        vLB = global_row_offsets_dev_ptr[rowOffsOffset + v];
        vUB = global_row_offsets_dev_ptr[rowOffsOffset + v + 1];
        for (int outgoingEdgeOfV = vLB + threadIdx.x; 
                outgoingEdgeOfV < vUB; 
                    outgoingEdgeOfV += blockDim.x){

                foundChild = myChild == global_columns_dev_ptr[valsAndColsOffset + outgoingEdgeOfV];
                // Set in-edge
                // store edge status
                tmp = global_values_dev_ptr[valsAndColsOffset + outgoingEdgeOfV];
                //   foundChild     tmp   (foundChild & tmp)  (foundChild & tmp)^tmp
                //1)      0          0            0                       0
                //2)      1          0            0                       0
                //3)      0          1            0                       1
                //4)      1          1            1                       0
                //
                // Case 1: isnt myChild and edge is off, stay off
                // Case 2: is myChild and edge is off, stay off
                // Case 3: isn't myChild and edge is on, stay on
                // Case 4: is myChild and edge is on, turn off
                // All this logic is necessary because we aren't using degree to set upperbound
                // we are using row offsets, which may include some edges turned off on a previous
                // pendant edge processing step.
                global_values_dev_ptr[valsAndColsOffset + outgoingEdgeOfV] ^= (foundChild & tmp);
        
        }
    }
    __syncthreads();
}

/* 
    Shared Memory Layout:
    1) DFS Paths of Length 4 = blockDim.x * 4
    2) Adjacency matrix = blockDim.x * blockDim.x
    3) Random Numbers = blockDim.x
    4) Pendant Path Boolean = blockDim.x
    4) Neighbors with a pendant boolean = blockDim.x
    5) Reduction buffer = blockDim.x
    6) I Set = blockDim.x
    7) V Set = blockDim.x
+ ______________________________________
    blockDim ^ 2 + 10*blockDim
*/

// SIMPLIFY THIS METHOD!
__global__ void ParallelIdentifyVertexDisjointNonPendantPaths(
                            
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_pendant_path_bool_dev_ptr,
                            int * global_paths_ptr,
                            int * global_set_inclusion_bool_ptr,
                            int * global_reduced_set_inclusion_count_ptr){
    //if (threadIdx.x == 0){
    printf("Block ID %d Started ParallelIdentifyVertexDisjointNonPendantPaths\n", blockIdx.x);
    //}

    int leafIndex = blockIdx.x;
    int globalPathOffset = leafIndex * 4 * blockDim.x;
    // Only allocated for one level, not tree global
    int globalPendantPathBoolOffset = blockIdx.x * blockDim.x;
    int globalSetInclusionBoolOffset = blockIdx.x * blockDim.x;

    extern __shared__ int pathsAndIndependentStatus[];

    int adjMatrixOffset = blockDim.x * 4;
    int randNumOffset = adjMatrixOffset + blockDim.x * blockDim.x;
    int pendPathBoolOffset = randNumOffset + blockDim.x;
    int neighborsWithAPendantOffset = pendPathBoolOffset + blockDim.x;
    int setReductionOffset = neighborsWithAPendantOffset + blockDim.x;
    int setInclusionOffset = setReductionOffset + blockDim.x;
    int setRemainingOffset = setInclusionOffset + blockDim.x;

    if (threadIdx.x == 0){
        printf("Block ID %d globalPathOffset %d\n", blockIdx.x, globalPathOffset);
        printf("Block ID %d globalPendantPathBoolOffset %d\n", blockIdx.x, globalPendantPathBoolOffset);
        printf("Block ID %d globalSetInclusionBoolOffset %d\n", blockIdx.x, globalSetInclusionBoolOffset);
        printf("Block ID %d adjMatrixOffset %d\n", blockIdx.x, adjMatrixOffset);
        printf("Block ID %d randNumOffset %d\n", blockIdx.x, randNumOffset);
        printf("Block ID %d pendPathBoolOffset %d\n", blockIdx.x, pendPathBoolOffset);
        printf("Block ID %d neighborsWithAPendantOffset %d\n", blockIdx.x, neighborsWithAPendantOffset);
        printf("Block ID %d setReductionOffset %d\n", blockIdx.x, setReductionOffset);
        printf("Block ID %d setInclusionOffset %d\n", blockIdx.x, setInclusionOffset);
        printf("Block ID %d setRemainingOffset %d\n", blockIdx.x, setRemainingOffset);
    }
    __syncthreads();

    // Write all 32 nonpendant paths to shared memory
    for (int start = threadIdx.x; start < blockDim.x*4; start += blockDim.x){
        pathsAndIndependentStatus[start] = global_paths_ptr[globalPathOffset + start];
    }
    if (threadIdx.x == 0){
        printf("Block ID %d threadIdx.x %d copied path into sm\n", blockIdx.x, threadIdx.x);
    }
    printf("Block ID %d path %d %s pendant\n", blockIdx.x, threadIdx.x, 
        global_pendant_path_bool_dev_ptr[globalPendantPathBoolOffset + threadIdx.x] ? "is" : "isn't");

    // Automatically include pendant  paths to set
    pathsAndIndependentStatus[pendPathBoolOffset + threadIdx.x] = 
        global_pendant_path_bool_dev_ptr[globalPendantPathBoolOffset + threadIdx.x];

    __syncthreads();
    if (threadIdx.x == 0){
        printf("Block ID %d threadIdx.x %d copied into sm\n", blockIdx.x, threadIdx.x);
    }
    // See if each vertex in my path is duplicated, 1 vs all comparison written to shared memory
    // Also, if it is duplicated, only process the largest index duplicate
    // If it isn't duplicated, process the path.

    // I need a for loop to define who "my path" is TPB times
    // my v1 versus 31 comparator v1's
    // my v2 versus 31 comparator v1's
    // my v3 versus 31 comparator v1's
    // my v4 versus 31 comparator v1's
    //              .
    //              .
    // my v1 versus 31 comparator v4's
    // my v2 versus 31 comparator v4's
    // my v3 versus 31 comparator v4's
    // my v4 versus 31 comparator v4's
    // myChild               comparatorChild 
    // ____________________________________
    // vertex % 4             vertex / 4
    //int myPathIndex = blockIdx.x % blockDim.x;
    printf("Block ID %d thread %d about to start adj mat\n", blockIdx.x, threadIdx.x);
    __syncthreads();

    int row, rowOffset, myChild, comparatorChild, vertex, myPathOffset, comparatorPathOffset;
    for (row = 0; row < blockDim.x; ++row){
        // blockDim.x*4 +  -- to skip the paths
        // the adj matrix size (32x32)
        rowOffset = adjMatrixOffset + row * blockDim.x;
        myPathOffset = row * 4;
        comparatorPathOffset = threadIdx.x * 4;
        printf("Block ID %d row %d started\n", blockIdx.x, row);
        for (vertex = 0; vertex < 4*4; ++vertex){
            // Same path for all TPB threads
            myChild = pathsAndIndependentStatus[myPathOffset + vertex / 4];
            // Different comparator child for all TPB threads
            comparatorChild = pathsAndIndependentStatus[comparatorPathOffset + vertex % 4];
            // Guarunteed to be true at least once, when i == j in adj matrix
            // We have a diagonal of ones.
            pathsAndIndependentStatus[rowOffset + threadIdx.x] |= (comparatorChild == myChild);
        }
        __syncthreads();
        printf("Block ID %d row %d done\n", blockIdx.x, row);
    }
    __syncthreads();
    if (threadIdx.x == 0){
        printf("Block ID %d threadIdx.x %d created adj matrix\n", blockIdx.x, threadIdx.x);
    }
    // Corresponds to an array of random numbers between [0,1]
    // This way every thread has its own randGen, and no thread sync is neccessary.
    unsigned int seed = 0;
    RNG::ctr_type r;
    r =  randomGPU_four(threadIdx.x, leafIndex, seed); 
    pathsAndIndependentStatus[randNumOffset + threadIdx.x] = r[0];
    __syncthreads();
     
    if (threadIdx.x == 0){
        for (int row = 0; row < blockDim.x; ++row){
            printf("Block ID %d threadIdx.x %d rand num %d\n", blockIdx.x, row, pathsAndIndependentStatus[randNumOffset + row]);
        }
    }
    for (int row = 0; row < blockDim.x; ++row){
        rowOffset = adjMatrixOffset + row*blockDim.x;
        // Check if any of my neighbors are pendant paths.
        // If I have a pendant neighbor I won't ever be included in the set.
        // Notably, the diagonal is true if the vertex is pendant
        // At this point the set I is the pendant paths
        pathsAndIndependentStatus[setReductionOffset + threadIdx.x] = pathsAndIndependentStatus[rowOffset + threadIdx.x]   
                                                                    && pathsAndIndependentStatus[pendPathBoolOffset + threadIdx.x];
        int i = blockDim.x/2;
        __syncthreads();
        while (i != 0) {
            if (threadIdx.x < i){
                pathsAndIndependentStatus[setReductionOffset + threadIdx.x] |= pathsAndIndependentStatus[setReductionOffset + threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }
        __syncthreads();
        if (threadIdx.x == 0){
            printf("Block ID %d row %d %s neighbors with a pendant edge\n", blockIdx.x, row, 
                pathsAndIndependentStatus[setReductionOffset] ? "is" :  "isn't");
            pathsAndIndependentStatus[neighborsWithAPendantOffset + row] = pathsAndIndependentStatus[setReductionOffset];
            // If it is neighbors (is) a pendant - false, it is not remaining; else - true
            pathsAndIndependentStatus[setRemainingOffset + row] = !pathsAndIndependentStatus[neighborsWithAPendantOffset + row];
            printf("Block ID %d row %d %s remaining in V\n", blockIdx.x, row, 
                pathsAndIndependentStatus[setRemainingOffset + row] ? "is" :  "isn't");
        }              
        __syncthreads();       
    }
    /*
    if (threadIdx.x == 0){
        printf("Adj Mat\n");
        for (int row = 0; row < blockDim.x; ++row){
            for(int colIndex = 0; colIndex < blockDim.x; ++colIndex){
                rowOffset = adjMatrixOffset + row*blockDim.x;
                printf("%d ", pathsAndIndependentStatus[rowOffset + colIndex]);
            }
            printf("\n");
        }
    }
    */
    // S = {p | p is a set of length 4 of vertex indices in G}
    // An edge (u,v), where u ∈ S, v ∈ S, and u ∩ v ≠ ∅
    // At this point I = {∀p ∈ S | p is pendant}
    // V = S / N(I)

    int cardinalityOfV = 1;

    // https://en.wikipedia.org/wiki/Maximal_independent_set#:~:text=Random-priority%20parallel%20algorithm%5Bedit%5D
    // Note that in every step, the node with the smallest number in each connected component always enters I, 
    // so there is always some progress. In particular, in the worst-case of the previous algorithm (n/2 connected components with 2 nodes each), a MIS will be found in a single step.
    // O(log_{4/3}(m)+1)
    while(cardinalityOfV){
        for (int row = 0; row < blockDim.x; ++row){
            // If a neighboring vertex has a random number less than mine and said vertex isn't
            // neighbors with a pendant edge, it should be added to the set, not me.
            pathsAndIndependentStatus[setReductionOffset + threadIdx.x] = pathsAndIndependentStatus[adjMatrixOffset + row*blockDim.x + threadIdx.x]
                                                                        && pathsAndIndependentStatus[setRemainingOffset + threadIdx.x]
                                                                        && (pathsAndIndependentStatus[randNumOffset + threadIdx.x]
                                                                            < pathsAndIndependentStatus[randNumOffset + row]);
            //printf("Row %d thread %d included %d\n", row, threadIdx.x, pathsAndIndependentStatus[setReductionOffset + threadIdx.x]);
            int i = blockDim.x/2;
            __syncthreads();
            while (i != 0) {
                if (threadIdx.x < i){
                    pathsAndIndependentStatus[setReductionOffset + threadIdx.x] |= pathsAndIndependentStatus[setReductionOffset + threadIdx.x + i];
                }
                __syncthreads();
                i /= 2;
            }
            __syncthreads();

            // If there exists a remaining vertex that has a smaller number than me, 
            // this reduces is true, thus when we negate it, it has no effect when or'ed
            // else if it was false, then it didn't fail, is negated to true and includes 
            // this vertex.
            // If this vertex was removed previously, we need to make sure it isn't added to I
            // Hence,  && pathsAndIndependentStatus[setRemainingOffset + row];
            if (threadIdx.x == 0){
                pathsAndIndependentStatus[setInclusionOffset + row] |= !pathsAndIndependentStatus[setReductionOffset] && pathsAndIndependentStatus[setRemainingOffset + row];
            }    
            __syncthreads();
            if (threadIdx.x == 0){
                printf("Block ID %d row %d %s included in the I set\n", blockIdx.x, row, 
                    pathsAndIndependentStatus[setInclusionOffset + row] ? "is" :  "isn't");
            }
        }

        for (int row = 0; row < blockDim.x; ++row){
            // Do we share an edge and were you included
            pathsAndIndependentStatus[setReductionOffset + threadIdx.x] = pathsAndIndependentStatus[adjMatrixOffset + row*blockDim.x + threadIdx.x]
                                                                        && pathsAndIndependentStatus[setInclusionOffset + threadIdx.x];
            int i = blockDim.x/2;
            __syncthreads();
            while (i != 0) {
                if (threadIdx.x < i){
                    pathsAndIndependentStatus[setReductionOffset + threadIdx.x] |= pathsAndIndependentStatus[setReductionOffset + threadIdx.x + i];
                }
                __syncthreads();
                i /= 2;
            }
            // If, so I will turn myself off
            // If I was included, I be included and I have an edge with myself, 
            // therefore I will be removed from V.
            if (threadIdx.x == 0){
                pathsAndIndependentStatus[setRemainingOffset + row] &= !pathsAndIndependentStatus[setReductionOffset];
            }    
            if (threadIdx.x == 0){
                printf("Block ID %d row %d %s remaining the V set\n", blockIdx.x, row, 
                    pathsAndIndependentStatus[setRemainingOffset + row] ? "is" :  "isn't");
            }
            __syncthreads();
        }

        // Am I remaining?
        pathsAndIndependentStatus[setReductionOffset + threadIdx.x] = pathsAndIndependentStatus[setRemainingOffset + threadIdx.x];
        int i = blockDim.x/2;
        __syncthreads();
        while (i != 0) {
            if (threadIdx.x < i){
                pathsAndIndependentStatus[setReductionOffset + threadIdx.x] += pathsAndIndependentStatus[setReductionOffset + threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }
        // when V is empty the algorithm terminates
        cardinalityOfV = pathsAndIndependentStatus[setReductionOffset];
        __syncthreads();
        
        if (threadIdx.x == 0){
            printf("Block ID %d cardinality of the V set is %d\n", blockIdx.x, cardinalityOfV);
        }
    }
    // Everything works to this point :)
    
    // We only have use for the non-pendant members of I, 
    // since the pendant paths have been processed already
    pathsAndIndependentStatus[setReductionOffset + threadIdx.x] = pathsAndIndependentStatus[setInclusionOffset + threadIdx.x]
        && !pathsAndIndependentStatus[pendPathBoolOffset + threadIdx.x];
    int i = blockDim.x/2;
    __syncthreads();
    while (i != 0) {
        if (threadIdx.x < i){
            pathsAndIndependentStatus[setReductionOffset + threadIdx.x] += pathsAndIndependentStatus[setReductionOffset + threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    __syncthreads();

    // Copy from shared mem to global..
    // We only have use for the non-pendant members of I, 
    // since the pendant paths have been processed already
    global_set_inclusion_bool_ptr[globalSetInclusionBoolOffset + threadIdx.x] = pathsAndIndependentStatus[setInclusionOffset + threadIdx.x]
        && !pathsAndIndependentStatus[pendPathBoolOffset + threadIdx.x];

    printf("Block ID %d row %d %s included in the I set\n", blockIdx.x, threadIdx.x, 
        pathsAndIndependentStatus[setInclusionOffset + threadIdx.x]
    && !pathsAndIndependentStatus[pendPathBoolOffset + threadIdx.x] ? "is" :  "isn't"); 

    // Record how many sets are in the MIS
    if (threadIdx.x == 0){
        global_reduced_set_inclusion_count_ptr[leafIndex] = pathsAndIndependentStatus[setReductionOffset];
    }          
}


/* 
    Shared Memory Layout:
    1) Reduction array
    2) DFS Paths of Length 4 = blockDim.x * 4
    3) Random Numbers = blockDim.x
    4) Pendant Path Boolean = blockDim.x
    5) Pendant Path Child = blockDim.x
    6) V Set = blockDim.x
    7) I Set = blockDim.x
+ ______________________________________
    10*blockDim
*/
__global__ void ParallelIdentifyVertexDisjointNonPendantPathsClean(
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_pendant_path_bool_dev_ptr,
                            int * global_pendant_child_dev_ptr,
                            int * global_paths_ptr,
                            int * global_set_inclusion_bool_ptr,
                            int * global_reduced_set_inclusion_count_ptr,
                            int * global_edges_left_to_cover_count,
                            int * global_verts_remain_count){
    if (threadIdx.x == 0){
        printf("Block ID %d Started ParallelIdentifyVertexDisjointNonPendantPathsClean\n", blockIdx.x);
    }
    int leafIndex = blockIdx.x;
    if (0 == global_edges_left_to_cover_count[leafIndex] || 0 == global_verts_remain_count[leafIndex]){
        global_reduced_set_inclusion_count_ptr[leafIndex] = 0;
        return;
    }

    int globalPathOffset = leafIndex * 4 * blockDim.x;
    // Only allocated for one level, not tree global
    int globalPendantPathBoolOffset = blockIdx.x * blockDim.x;
    int globalPendantPathChildOffset = globalPendantPathBoolOffset;
    int globalSetInclusionBoolOffset = globalPendantPathBoolOffset;

    extern __shared__ int pathsAndIndependentStatus[];

    int setReductionOffset = 0;
    int pathsOffset = blockDim.x;
    int randNumOffset = pathsOffset + blockDim.x * 4;
    int pendPathBoolOffset = randNumOffset + blockDim.x;
    int pendPathChildOffset = pendPathBoolOffset + blockDim.x;
    int setRemainingOffset = pendPathChildOffset + blockDim.x;
    int setInclusionOffset = setRemainingOffset + blockDim.x;

    // Write all 32 nonpendant paths to shared memory
    for (int start = threadIdx.x; start < blockDim.x*4; start += blockDim.x){
        pathsAndIndependentStatus[pathsOffset + start] = global_paths_ptr[globalPathOffset + start];
    }

    // Write all 32 pendant booleans to shared memory
    for (int start = threadIdx.x; start < blockDim.x; start += blockDim.x){
        pathsAndIndependentStatus[pendPathBoolOffset + start] = global_pendant_path_bool_dev_ptr[globalPendantPathBoolOffset + start];
    }

    // Write all 32 pendant children to shared memory
    for (int start = threadIdx.x; start < blockDim.x; start += blockDim.x){
        pathsAndIndependentStatus[pendPathChildOffset + start] = global_pendant_child_dev_ptr[globalPendantPathChildOffset + start];
    }

    // Initialize I set to 0
    for (int start = threadIdx.x; start < blockDim.x; start += blockDim.x){
        pathsAndIndependentStatus[setInclusionOffset + start] = 0;
    }
    __syncthreads();

    // Automatically remove paths containing the pendant child(ren)
    // 1 path vs all pendant children comparison of each row.  Duplicate pendant children
    // aren't a problem.
    int row, myChild, pendantChild, vertex, myPathOffset, pendantBool;
    for (row = 0; row < blockDim.x; ++row){
        myPathOffset = pathsOffset + threadIdx.x * 4;
        pathsAndIndependentStatus[setReductionOffset + threadIdx.x] = 0;
        pendantBool = pathsAndIndependentStatus[pendPathBoolOffset + row];
        pendantChild = pathsAndIndependentStatus[pendPathChildOffset + row];
        for (vertex = 0; vertex < 4; ++vertex){
            // Same path for all TPB threads
            myChild = pathsAndIndependentStatus[myPathOffset + vertex];
            // Different comparator child for all TPB threads
            // Guarunteed to be true at least once, when i == j in adj matrix
            // We have a diagonal of ones.
            pathsAndIndependentStatus[setReductionOffset + threadIdx.x] |= (pendantChild == myChild) && pendantBool;
        }
        __syncthreads();
        int i = blockDim.x/2;
        while (i != 0) {
            if (threadIdx.x < i){
                pathsAndIndependentStatus[setReductionOffset + threadIdx.x] |= pathsAndIndependentStatus[setReductionOffset + threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }
        if(threadIdx.x == 0)
            pathsAndIndependentStatus[setRemainingOffset + row] = !pathsAndIndependentStatus[setReductionOffset];
        __syncthreads();
    }

    // See if each vertex in my path is duplicated, 1 vs all comparison written to shared memory
    // Also, if it is duplicated, only process the largest index duplicate
    // If it isn't duplicated, process the path.

    // I need a for loop to define who "my path" is TPB times
    // my v1 versus 31 comparator v1's
    // my v2 versus 31 comparator v1's
    // my v3 versus 31 comparator v1's
    // my v4 versus 31 comparator v1's
    //              .
    //              .
    // my v1 versus 31 comparator v4's
    // my v2 versus 31 comparator v4's
    // my v3 versus 31 comparator v4's
    // my v4 versus 31 comparator v4's
    // myChild               comparatorChild 
    // ____________________________________
    // vertex % 4             vertex / 4
    //int myPathIndex = blockIdx.x % blockDim.x;

    // S = {p | p is a set of length 4 of vertex indices in G}
    // An edge (u,v), where u ∈ S, v ∈ S, and u ∩ v ≠ ∅
    // At this point I = {∀p ∈ S | p is pendant}
    // V = S / N(I)

    int cardinalityOfV = 1;
    unsigned int seed = 0;

    // https://en.wikipedia.org/wiki/Maximal_independent_set#:~:text=Random-priority%20parallel%20algorithm%5Bedit%5D
    // Note that in every step, the node with the smallest number in each connected component always enters I, 
    // so there is always some progress. In particular, in the worst-case of the previous algorithm (n/2 connected components with 2 nodes each), a MIS will be found in a single step.
    // O(log_{4/3}(m)+1)
    while(cardinalityOfV){
        // Corresponds to an array of random numbers between [0,1]
        // This way every thread has its own randGen, and no thread sync is neccessary.
        RNG::ctr_type r;
        r =  randomGPU_four(threadIdx.x, leafIndex, seed); 
        pathsAndIndependentStatus[randNumOffset + threadIdx.x] = r[0];
        __syncthreads();


        // I can't change setRemaining within this for loop!
        int comparatorChild, comparatorPathOffset;
        for (row = 0; row < blockDim.x; ++row){
            pathsAndIndependentStatus[setReductionOffset + threadIdx.x] = 0;
            myPathOffset = pathsOffset + row * 4;
            comparatorPathOffset = pathsOffset + threadIdx.x * 4;
            // Does an edge exist between these paths?
            for (vertex = 0; vertex < 4*4; ++vertex){
                // Same path for all TPB threads
                myChild = pathsAndIndependentStatus[myPathOffset + vertex / 4];
                // Different comparator child for all TPB threads
                comparatorChild = pathsAndIndependentStatus[comparatorPathOffset + vertex % 4];
                // Guarunteed to be true at least once, when i == j in adj matrix
                // We have a diagonal of ones.
                pathsAndIndependentStatus[setReductionOffset + threadIdx.x] |= (comparatorChild == myChild);
            }
            pathsAndIndependentStatus[setReductionOffset + threadIdx.x] &= (pathsAndIndependentStatus[setRemainingOffset + threadIdx.x]
                                                                            && pathsAndIndependentStatus[setRemainingOffset + row]);
            // In the case I have no edges, I still share an edge with myself.  Then I am not less than myself.
            // and I am remaining, therefore I will be included in the I set and removed from the V set.
            pathsAndIndependentStatus[setReductionOffset + threadIdx.x] &= (pathsAndIndependentStatus[randNumOffset + threadIdx.x]
                                                                            < pathsAndIndependentStatus[randNumOffset + row]);
            __syncthreads();

            int i = blockDim.x/2;
            while (i != 0) {
                if (threadIdx.x < i){
                    pathsAndIndependentStatus[setReductionOffset + threadIdx.x] |= pathsAndIndependentStatus[setReductionOffset + threadIdx.x + i];
                }
                __syncthreads();
                i /= 2;
            }

            // Did any of my neighbors make it into the I set?
            // If not, am I remaining?
            // If so, add myself to the inclusion set.
            if(threadIdx.x == 0)
                pathsAndIndependentStatus[setInclusionOffset + row] |= !pathsAndIndependentStatus[setReductionOffset] && pathsAndIndependentStatus[setRemainingOffset + row];
            __syncthreads();
        }

        // Remove all neighbors of Included vertices from setRemaining.
        for (row = 0; row < blockDim.x; ++row){
            pathsAndIndependentStatus[setReductionOffset + threadIdx.x] = 0;
            myPathOffset = pathsOffset + row * 4;
            comparatorPathOffset = pathsOffset + threadIdx.x * 4;
            // Does an edge exist between these paths?
            for (vertex = 0; vertex < 4*4; ++vertex){
                // Same path for all TPB threads
                myChild = pathsAndIndependentStatus[myPathOffset + vertex / 4];
                // Different comparator child for all TPB threads
                comparatorChild = pathsAndIndependentStatus[comparatorPathOffset + vertex % 4];
                // Guarunteed to be true at least once, when i == j in adj matrix
                // We have a diagonal of ones.
                pathsAndIndependentStatus[setReductionOffset + threadIdx.x] |= (comparatorChild == myChild);
            }
            // Are we neightbors? Whether I or you are remaining is irrelevant.
            // If so, was I included in the inclusion set?
            // If so, remove you from the remaining set.
            pathsAndIndependentStatus[setRemainingOffset + threadIdx.x] &= !(pathsAndIndependentStatus[setReductionOffset + threadIdx.x] 
                                                                        &&  pathsAndIndependentStatus[setInclusionOffset + row]);
            __syncthreads();
        }

        pathsAndIndependentStatus[setReductionOffset + threadIdx.x] = pathsAndIndependentStatus[setRemainingOffset + threadIdx.x];
        __syncthreads();
        int i = blockDim.x/2;
        while (i != 0) {
            if (threadIdx.x < i){
                pathsAndIndependentStatus[setReductionOffset + threadIdx.x] += pathsAndIndependentStatus[setReductionOffset + threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }

        cardinalityOfV = pathsAndIndependentStatus[setReductionOffset];
        ++seed;
    }
    pathsAndIndependentStatus[setReductionOffset + threadIdx.x] = pathsAndIndependentStatus[setInclusionOffset + threadIdx.x];
    int i = blockDim.x/2;
    __syncthreads();
    while (i != 0) {
        if (threadIdx.x < i){
            pathsAndIndependentStatus[setReductionOffset + threadIdx.x] += pathsAndIndependentStatus[setReductionOffset + threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    __syncthreads();

    // Record how many sets are in the MIS
    global_set_inclusion_bool_ptr[globalSetInclusionBoolOffset + threadIdx.x] = pathsAndIndependentStatus[setInclusionOffset + threadIdx.x];
    if (threadIdx.x == 0){
        global_reduced_set_inclusion_count_ptr[leafIndex] = pathsAndIndependentStatus[setReductionOffset];
    }          
}

    //int leafValue = global_active_leaf_value[leafIndex];
    // Solve recurrence relation 
    // g(n) = 1/6*((2*C+3)*3^n - 3)
    // C depends on leafValue
    // where g(0) = left-most child of depth 1
    // where g(1) = left-most child of depth 2
    // where g(2) = left-most child of depth 3
    // ...
    //int arbitraryParameter = 3*(3*leafValue)+1));

__global__ void ParallelAssignMISToNodesBreadthFirstClean(int * global_active_leaf_value,
                                        int * global_set_paths_indices,
                                        int * global_reduced_set_inclusion_count_ptr,
                                        int * global_paths_ptr,
                                        int * global_vertices_included_dev_ptr,
                                        int * global_edges_left_to_cover_count,
                                        int * global_verts_remain_count){
    int leafIndex = blockIdx.x;
    if (0 == global_edges_left_to_cover_count[leafIndex] || 0 == global_verts_remain_count[leafIndex])
        return;
    int leafValue = global_active_leaf_value[leafIndex];
    // Solve recurrence relation 
    // g(n) = 1/6*((2*C+3)*3^n - 3)
    // C depends on leafValue
    // where g(0) = left-most child of depth 1
    // where g(1) = left-most child of depth 2
    // where g(2) = left-most child of depth 3
    // ... 
    int arbitraryParameter = 3*((3*leafValue)+1);
    int leftMostChildOfLevel;
    int leftMostChildOfLevelExpanded;
    int setPathOffset = leafIndex * blockDim.x;
    int globalPathOffset = setPathOffset*4;
    extern __shared__ int paths[];

    // |I| - The cardinality of the set.  If |I| = 0; we don't induce children
    // Else we will induce (3*|I| children), Each path induces 3 leaves.
    int leavesThatICanProcess = global_reduced_set_inclusion_count_ptr[leafIndex];
    // Note: I am sorting the indices into the paths by inclusion in the set
    // I can achieve coalescence by actually rearranging the paths.  The problem is 
    // the cub library doesnt seem to support soring by key, when the value is
    // 4 ints and the key being 1 int.  I will look further into this.

    // leavesThatICanProcess is necessarily < tPB
    for (int index = threadIdx.x; index < leavesThatICanProcess; index += blockDim.x){
        paths[index] = global_set_paths_indices[setPathOffset + index];
    }
    __syncthreads();
    int pathIndex, pathValue, pathChild;
    for (int index = threadIdx.x; index < leavesThatICanProcess*4; index += blockDim.x){
        pathIndex = index / 4;
        pathChild = index % 4;
        pathValue = global_set_paths_indices[setPathOffset + pathIndex];
        paths[blockDim.x + index] = global_paths_ptr[globalPathOffset + pathValue*4 + pathChild];
    }    
    __syncthreads();

    if (threadIdx.x == 0){
        printf("Block ID %d thread %d entered ParallelAssignMISToNodesBreadthFirst\n", blockIdx.x, threadIdx.x);
        printf("Block ID %d thread %d can process leafIndex %d\n", blockIdx.x, threadIdx.x, leafIndex);
        printf("Block ID %d thread %d can process leafValue %d\n", blockIdx.x, threadIdx.x, leafValue);
        printf("Block ID %d thread %d can process %d leaves\n", blockIdx.x, threadIdx.x, leavesThatICanProcess);
    }
    // This pattern uses adjacent threads to write aligned memory, 
    // but thread indexing is math intensive
    // Desired mapping:
    // 0 -> 2 
    // 1 -> 0
    
    // 2 -> 2
    // 3 -> 1
    // 4 -> 3
    // 5 -> 1
    // indexMod6 = index % 6
    // Functor: (indexMod6 % 2 == 1) * (indexMod6 != 1) +
    //          (indexMod6 % 2 == 0) * (2 + (indexMod6 == 4))

    int indexMod6, pathChildIndex, dispFromLeft, levelDepth, relativeLeafIndex, totalNodes;
    for(int index = threadIdx.x; index < leavesThatICanProcess*6; index += blockDim.x){
        pathIndex = index / 6;
        indexMod6 = index % 6;
        // This can be considered a function of leafValue and index ...
        // Have to handle 0 and 1..
        pathChildIndex = (indexMod6 % 2 == 1) * (indexMod6 != 1) +
                         (indexMod6 % 2 == 0) * (2 + (indexMod6 == 4));
        relativeLeafIndex = index/2;
        levelDepth = 0;
        // No closed form solution exists.
        //levelDepth += (int)(relativeLeafIndex / 1 != 0);
        // 0 + 3^1
        levelDepth += (int)(relativeLeafIndex / 3 != 0);
        // 3 + 3^2
        levelDepth += (int)(relativeLeafIndex / 12 != 0);
        // 12 + 3^3
        levelDepth += (int)(relativeLeafIndex / 39 != 0);
        // 39 + 3^4
        levelDepth += (int)(relativeLeafIndex / 120 != 0);
        // 120 + 3^5
        levelDepth += (int)(relativeLeafIndex / 363 != 0);
        // 363 + 3^6
        levelDepth += (int)(relativeLeafIndex / 1092 != 0);
        // 1092 + 3^7
        levelDepth += (int)(relativeLeafIndex / 3279 != 0);
        // Closed form solution of recurrence relation shown in comment above method
        leftMostChildOfLevel = ((2*arbitraryParameter+3)*powf(3.0, levelDepth) - 3)/6;
        leftMostChildOfLevelExpanded = 2*leftMostChildOfLevel-1;
        // Closed form sum of Geometric Series 3^k
        totalNodes = (levelDepth!=0)*(((1.0-powf(3.0, levelDepth+1))/(1.0-3.0))-1);
        dispFromLeft = index - 2*totalNodes;
        /*
        if (blockIdx.x == 0){
            printf("thread %d index %d\n",threadIdx.x, index);
            printf("thread %d pathIndex %d\n", threadIdx.x, pathIndex);
            printf("thread %d indexMod6 %d\n", threadIdx.x, indexMod6);
            printf("thread %d pathChildIndex %d\n", threadIdx.x, pathChildIndex);
            printf("thread %d levelDepth %d\n", threadIdx.x, levelDepth);
            printf("thread %d leftMostChildOfLevel %d\n", threadIdx.x, leftMostChildOfLevel);
            printf("thread %d leftMostChildOfLevelExpanded %d\n", threadIdx.x, leftMostChildOfLevelExpanded);
            printf("thread %d dispFromLeft %d\n", threadIdx.x, dispFromLeft);
        }
        */
        global_vertices_included_dev_ptr[leftMostChildOfLevelExpanded + dispFromLeft] = paths[blockDim.x + pathIndex*4 + pathChildIndex];
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("VertsIncluded\n");
        int numLvls = 4;
        int LB = 0, UB = 0;
        for (int lvl = 0; lvl < numLvls; ++lvl){
            if (LB == 0)
                UB = 1;
            else
                UB = LB + (int)(powf(3.0, lvl)*2.0);
            //printf("LB : %d; UB : %d\n ", LB, UB);
            for (int i = LB; i < UB; ++i){
                printf("%d ", global_vertices_included_dev_ptr[i]);
            }
            printf("\n");
            if (LB == 0)
                LB = 1;
            else
                LB = LB + (int)(powf(3.0, lvl)*2.0);
        }

    }
}

__global__ void ParallelAssignMISToNodesBreadthFirst(int * global_active_leaf_value,
                                        int * global_set_paths_indices,
                                        int * global_reduced_set_inclusion_count_ptr,
                                        int * global_paths_ptr,
                                        int * global_vertices_included_dev_ptr){
    int leafIndex = blockIdx.x;
    int leafValue = global_active_leaf_value[leafIndex];
    int setPathOffset = leafIndex * blockDim.x;
    int globalPathOffset = setPathOffset*4;
    // |I| - The cardinality of the set.  If |I| = 0; we don't induce children
    // Else we will induce (3*|I| children), Each path induces 3 leaves.
    int leavesThatICanProcess = global_reduced_set_inclusion_count_ptr[leafIndex];
    if (threadIdx.x == 0){
        printf("Block ID %d thread %d entered ParallelAssignMISToNodesBreadthFirst\n", blockIdx.x, threadIdx.x);
        printf("Block ID %d thread %d can process leafIndex %d\n", blockIdx.x, threadIdx.x, leafIndex);
        printf("Block ID %d thread %d can process leafValue %d\n", blockIdx.x, threadIdx.x, leafValue);
        printf("Block ID %d thread %d can process %d leaves\n", blockIdx.x, threadIdx.x, leavesThatICanProcess);
    }
    // This pattern uses adjacent threads to write aligned memory, 
    // but thread indexing is math intensive
    // Desired mapping:
    // 0 -> 2 
    // 1 -> 0
    // 2 -> 2
    // 3 -> 1
    // 4 -> 3
    // 5 -> 1
    // indexMod6 = index % 6
    // Functor: (indexMod6 % 2 == 1) * (indexMod6 != 1) +
    //          (indexMod6 % 2 == 0) * (2 + (indexMod6 == 4))

    int indexMod6, pathChildIndex, pathIndex, pathValue, leftMostChildOfLevel, leftMostChildOfLevelExpanded, dispFromLeft, levelDepth, indexMapper, levelWidth;
    for(int index = threadIdx.x; index < leavesThatICanProcess*6; index += blockDim.x){
        pathIndex = index / 6;
        pathValue = global_set_paths_indices[setPathOffset + pathIndex];
        indexMod6 = index % 6;
        // Have to handle 0 and 1..
        pathChildIndex = (indexMod6 % 2 == 1) * (indexMod6 != 1) +
                            (indexMod6 % 2 == 0) * (2 + (indexMod6 == 4));
        levelDepth = 1.0;
        indexMapper = index;
        leftMostChildOfLevel = leafValue + (leafValue == 0);
        levelWidth = (int)(2.0*powf(3.0, levelDepth));
        leftMostChildOfLevelExpanded = 0;
        while(indexMapper / levelWidth){
            indexMapper -=  (int)(2*powf(3.0, levelDepth));
            ++levelDepth;
            leftMostChildOfLevel *= 3.0;
            leftMostChildOfLevelExpanded += leftMostChildOfLevel;
            levelWidth = (int)(2.0*powf(3.0, levelDepth));
            indexMapper = indexMapper*((int)(indexMapper >= 0));
        }
        // Handles index 0 : + (int)(leftMostChildOfLevelExpanded == 0)
        leftMostChildOfLevelExpanded = ((int)(leftMostChildOfLevelExpanded != 0))*(leftMostChildOfLevelExpanded*2 + 1) + (int)(leftMostChildOfLevelExpanded == 0);
        printf("thread %d levelWidth %d\n",threadIdx.x, levelWidth);
        dispFromLeft = index - leftMostChildOfLevelExpanded + 1;
        /*
        if (blockIdx.x == 0){
            printf("thread %d index %d\n",threadIdx.x, index);
            printf("thread %d pathIndex %d\n", threadIdx.x, pathIndex);
            printf("thread %d pathValue %d\n", threadIdx.x, pathValue);
            printf("thread %d indexMod6 %d\n", threadIdx.x, indexMod6);
            printf("thread %d pathChildIndex %d\n", threadIdx.x, pathChildIndex);
            printf("thread %d levelDepth %d\n", threadIdx.x, levelDepth);
            printf("thread %d leftMostChildOfLevel %d\n", threadIdx.x, leftMostChildOfLevel);
            printf("thread %d leftMostChildOfLevelExpanded %d\n", threadIdx.x, leftMostChildOfLevelExpanded);
            printf("thread %d dispFromLeft %d\n", threadIdx.x, dispFromLeft);
        }
        */
        global_vertices_included_dev_ptr[leftMostChildOfLevelExpanded + dispFromLeft] = global_paths_ptr[globalPathOffset + pathValue*4 + pathChildIndex];
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("VertsIncluded\n");
        int numLvls = 4;
        int LB = 0, UB = 0;
        for (int lvl = 0; lvl < numLvls; ++lvl){
            if (LB == 0)
                UB = 1;
            else
                UB = LB + (int)(powf(3.0, lvl)*2.0);
            //printf("LB : %d; UB : %d\n ", LB, UB);
            for (int i = LB; i < UB; ++i){
                printf("%d ", global_vertices_included_dev_ptr[i]);
            }
            printf("\n");
            if (LB == 0)
                LB = 1;
            else
                LB = LB + (int)(powf(3.0, lvl)*2.0);
        }

    }
}

__global__ void ParallelCalculateOffsetsForNewlyActivateLeafNodesBreadthFirst(
                                        int * global_active_leaves_count_current,
                                        int * global_active_leaves_count_new,
                                        int * global_reduced_set_inclusion_count_ptr,
                                        int * global_newly_active_leaves_count_ptr,
                                        int * global_edges_left_to_cover_count,
                                        int * global_verts_remain_count){
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ int new_active_leaves_count_red[];
    // We need to enter this loop to set leavesToProcess to 0
    // for terminating condition.
    if (globalIndex < global_active_leaves_count_current[0]){

        printf("globalIndex %d, global_active_leaves_count_current %d\n",globalIndex, global_active_leaves_count_current[0]);
        printf("globalIndex %d, ParallelCalculateOffsetsForNewlyActivateLeafNodesBreadthFirst\n",globalIndex);
        int leavesToProcess = global_reduced_set_inclusion_count_ptr[globalIndex];
        // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
        // Solved for leavesToProcess < closed form
        // start from level 1, hence add a level if LTP > 0, 1 complete level 
        // Add 1 if LTP == 0 to prevent runtime error
        // LTP = 2
        // CL = 1
                // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
        int completeLevel = floor(logf(2*leavesToProcess + 1) / logf(3)) - (int)(leavesToProcess==0);
        // If LTP == 0, we dont want to create any new leaves
        // Therefore, we dont want to enter the for loops.
        // The active leaf writes itself as it's parent before the for loops
        // This is overwritten within the for loops if LTP > 0
        // CLL = 3
        int leavesFromCompleteLvl = powf(3.0, completeLevel) - (int)(leavesToProcess == 0);
        // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
        // Solved for closed form < leavesToProcess
        // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
        // IL = 1
        int incompleteLevel = ceil(logf(2*leavesToProcess + 1) / logf(3)) - (int)(leavesToProcess==0);
        // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
        // Add 1 when leavesToProcess isn't 0, so we start counting from level 1
        // Also subtract the root, so we start counting from level 1
        // TSC = 3
        int treeSizeComplete = (1.0 - powf(3.0, completeLevel+(int)(leavesToProcess != 0)))/(1.0 - 3.0) - (int)(leavesToProcess != 0);
        // How many internal leaves to skip in complete level
        // RFC = 1
        int removeFromComplete = ((3*leavesToProcess - treeSizeComplete) + 3 - 1) / 3;
        // Leaves that are used in next level
        int leavesFromIncompleteLvl = 3*removeFromComplete;
        printf("Leaves %d, completeLevel Depth %d\n",leavesToProcess, completeLevel);
        printf("Leaves %d, leavesFromCompleteLvl %d\n",leavesToProcess, leavesFromCompleteLvl);
        printf("Leaves %d, incompleteLevel Depth %d\n",leavesToProcess, incompleteLevel);
        printf("Leaves %d, treeSizeComplete Leaves%d\n",leavesToProcess, treeSizeComplete);
     
        printf("Leaves %d, removeFromComplete %d\n",leavesToProcess, removeFromComplete);
        int totalNewActive = (leavesFromCompleteLvl - removeFromComplete) + leavesFromIncompleteLvl;
        printf("Leaves %d, totalNewActive %d\n",leavesToProcess, totalNewActive);
        // Write to global memory
        // If new leaves == 0, then either the graph is empty, which will be handled elsewhere
        // Or every path was on a pendant node, and the current vertex should be written to the 
        // list of new active vertices.
        global_newly_active_leaves_count_ptr[globalIndex] = totalNewActive + (int)(totalNewActive == 0);
        // Write to shared memory for reduction
        new_active_leaves_count_red[threadIdx.x] = totalNewActive + (int)(totalNewActive == 0);
        // If no edges are left or no vertices are remaining, then deactivate this leaf.
        new_active_leaves_count_red[threadIdx.x] *= (int)(global_edges_left_to_cover_count[globalIndex] != 0);
        new_active_leaves_count_red[threadIdx.x] *= (int)(global_verts_remain_count[globalIndex] != 0);
    } else {
        new_active_leaves_count_red[threadIdx.x] = 0;
    }
    int i = blockDim.x/2;
    __syncthreads();
    while (i != 0) {
        if (threadIdx.x < i){
            printf("new_active_leaves_count_red[%d] = %d + %d\n", threadIdx.x, new_active_leaves_count_red[threadIdx.x], new_active_leaves_count_red[threadIdx.x + i]);
            new_active_leaves_count_red[threadIdx.x] += new_active_leaves_count_red[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0)
        atomicAdd(global_active_leaves_count_new, new_active_leaves_count_red[threadIdx.x]);
}

    //int leafValue = global_active_leaf_value[leafIndex];
    // Solve recurrence relation 
    // g(n) = 1/6*((2*C+3)*3^n - 3)
    // C depends on leafValue
    // where g(0) = left-most child of depth 1
    // where g(1) = left-most child of depth 2
    // where g(2) = left-most child of depth 3
    // ...
    //int arbitraryParameter = 3*(3*leafValue)+1);

// Not thrilled about this.  1 thread fills in all the entries of belonging to a single active leaf
// in the new active leaves buffer.  To avoid this I'd likely need another buffer
__global__ void ParallelPopulateNewlyActivateLeafNodesBreadthFirst(
                                        int * global_active_leaves,
                                        int * global_newly_active_leaves,
                                        int * global_active_leaves_count_current,
                                        int * global_reduced_set_inclusion_count_ptr,
                                        int * global_newly_active_offset_ptr,
                                        int * global_active_leaf_parent_leaf_index,
                                        int * global_active_leaf_parent_leaf_value){
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int leafValue;

    printf("globalIndex %d, global_active_leaves_count_current %x\n",globalIndex, global_active_leaves_count_current[0]);
    if (globalIndex < global_active_leaves_count_current[0]){
        int leavesToProcess = global_reduced_set_inclusion_count_ptr[globalIndex];
        // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
        // Solved for leavesToProcess < closed form
        int completeLevel = floor(logf(2*leavesToProcess + 1) / logf(3));
        int leavesFromCompleteLvl = powf(3.0, completeLevel);
        // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
        // Solved for closed form < leavesToProcess
        int incompleteLevel = ceil(logf(2*leavesToProcess + 1) / logf(3));
        // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
        int treeSizeComplete = (1.0 - powf(3.0, completeLevel))/(1.0 - 3.0);
        printf("Leaves %d, completeLevel Level Depth %d\n",leavesToProcess, completeLevel);
        printf("Leaves %d, leavesFromCompleteLvl Level Depth %d\n",leavesToProcess, leavesFromCompleteLvl);
        printf("Leaves %d, incompleteLevel Level Depth %d\n",leavesToProcess, incompleteLevel);
        printf("Leaves %d, treeSizeComplete Level Depth %d\n",leavesToProcess, treeSizeComplete);
        int removeFromComplete = ((leavesToProcess - treeSizeComplete) + 3 - 1) / 3;
        int leavesFromIncompleteLvl = (leavesToProcess - treeSizeComplete);
        leafValue = global_active_leaves[globalIndex];
        int leftMostLeafIndexOfFullLevel = leafValue;
        while (completeLevel > 0){
            leftMostLeafIndexOfFullLevel = (3.0 * leftMostLeafIndexOfFullLevel + 1);
            completeLevel -= 1;
        }
        int newly_active_offset = global_newly_active_offset_ptr[globalIndex];
        int index = 0;
        // These values will be overwritten in the for loops, if leavesToProcess > 0
        // Therefore initialize the values as if there were not any non-pendant paths
        // found in the DFS.  This way we minimize the amount of conditionals.
        global_newly_active_leaves[newly_active_offset + index] = leafValue;
        global_active_leaf_parent_leaf_value[newly_active_offset + index] = leafValue;
        global_active_leaf_parent_leaf_index[newly_active_offset + index] = globalIndex;

        // If non-pendant paths were found, populate the search tree in the 
        // complete level
        for (; index < leavesFromCompleteLvl - removeFromComplete; ++index){
            //printf("global_newly_active_leaves[%d] = %d\n",newly_active_offset + index, leftMostLeafIndexOfFullLevel + index + removeFromComplete);
            global_newly_active_leaves[newly_active_offset + index] = leftMostLeafIndexOfFullLevel + index + removeFromComplete;
            global_active_leaf_parent_leaf_value[newly_active_offset + index] = leafValue;
            global_active_leaf_parent_leaf_index[newly_active_offset + index] = globalIndex;
        }
        int leftMostLeafIndexOfIncompleteLevel = leafValue;
        while (incompleteLevel > 0){
            leftMostLeafIndexOfIncompleteLevel = (3.0 * leftMostLeafIndexOfIncompleteLevel + 1);
            incompleteLevel -= 1;
        }
        int totalNewActive = 3*leavesFromIncompleteLvl + leavesFromCompleteLvl - removeFromComplete;
        printf("Leaves %d, totalNewActive %d\n",leavesToProcess, totalNewActive);
        // If non-pendant paths were found, populate the search tree in the 
        // incomplete level
        for (int incompleteIndex = 0; index < totalNewActive; ++index, ++incompleteIndex){
            //printf("global_newly_active_leaves[%d] = %d\n",newly_active_offset + index, leftMostLeafIndexOfIncompleteLevel + incompleteIndex);
            global_newly_active_leaves[newly_active_offset + index] = leftMostLeafIndexOfIncompleteLevel + incompleteIndex;
            global_active_leaf_parent_leaf_value[newly_active_offset + index] = leafValue;
            global_active_leaf_parent_leaf_index[newly_active_offset + index] = globalIndex;
        }
        //for (int testP = 0; testP < global_active_leaves_count_new[0]; ++testP){
        //    printf("global_newly_active_leaves[%d] = %d\n",testP, global_newly_active_leaves[testP]);
        //}
    }
}

__global__ void ParallelPopulateNewlyActivateLeafNodesBreadthFirstClean(
                                        int * global_active_leaves,
                                        int * global_newly_active_leaves,
                                        int * global_active_leaves_count_current,
                                        int * global_reduced_set_inclusion_count_ptr,
                                        int * global_newly_active_offset_ptr,
                                        int * global_active_leaf_index,
                                        int * global_active_leaf_parent_leaf_index,
                                        int * global_active_leaf_parent_leaf_value,
                                        int * global_edges_left_to_cover_count,
                                        int * global_verts_remain_count){
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int leafValue;
    int arbitraryParameter;
    int leftMostLeafIndexOfFullLevel;
    int leftMostLeafIndexOfIncompleteLevel;
    // Since number of leaves is not necessarily a power of 2
    // Dont populate inactivated leaves.
    if (globalIndex < global_active_leaves_count_current[0] 
        && 0 < global_verts_remain_count[globalIndex]
        && 0 < global_edges_left_to_cover_count[globalIndex]){

        printf("globalIndex %d, ParallelPopulateNewlyActivateLeafNodesBreadthFirstClean\n",globalIndex);
        printf("globalIndex %d, global_active_leaves_count_current %x\n",globalIndex, global_active_leaves_count_current[0]);
        int leavesToProcess = global_reduced_set_inclusion_count_ptr[globalIndex];
        // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
        // Solved for leavesToProcess < closed form
        // start from level 1, hence add a level if LTP > 0, 1 complete level 
        // Add 1 if LTP == 0 to prevent runtime error
        // LTP = 2
        // CL = 1
        // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
        int completeLevel = floor(logf(2*leavesToProcess + 1) / logf(3)) - (int)(leavesToProcess==0);
        // If LTP == 0, we dont want to create any new leaves
        // Therefore, we dont want to enter the for loops.
        // The active leaf writes itself as it's parent before the for loops
        // This is overwritten within the for loops if LTP > 0
        // CLL = 3
        int leavesFromCompleteLvl = powf(3.0, completeLevel) - (int)(leavesToProcess == 0);
        // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
        // Solved for closed form < leavesToProcess
        // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
        // IL = 1
        int incompleteLevel = ceil(logf(2*leavesToProcess + 1) / logf(3)) - (int)(leavesToProcess==0);
        // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
        // Add 1 when leavesToProcess isn't 0, so we start counting from level 1
        // Also subtract the root, so we start counting from level 1
        // TSC = 3
        int treeSizeComplete = (1.0 - powf(3.0, completeLevel+(int)(leavesToProcess != 0)))/(1.0 - 3.0) - (int)(leavesToProcess != 0);
        // How many internal leaves to skip in complete level
        // RFC = 1
        int removeFromComplete = ((3*leavesToProcess - treeSizeComplete) + 3 - 1) / 3;
        // Leaves that are used in next level
        int leavesFromIncompleteLvl = 3*removeFromComplete;
        
        leafValue = global_active_leaves[globalIndex];
        arbitraryParameter = 3*((3*leafValue)+1);
        // Closed form solution of recurrence relation shown in comment above method
        // Subtract 1 because reasons
        leftMostLeafIndexOfFullLevel = ((2*arbitraryParameter+3)*powf(3.0, completeLevel-1) - 3)/6;
        leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, incompleteLevel-1) - 3)/6;

        int newly_active_offset = global_newly_active_offset_ptr[globalIndex];
        int index = 0;
        // These values will be overwritten in the for loops, if leavesToProcess > 0
        // Therefore initialize the values as if there were not any non-pendant paths
        // found in the DFS.  This way we minimize the amount of conditionals.
        global_newly_active_leaves[newly_active_offset + index] = leafValue;
        global_active_leaf_parent_leaf_value[newly_active_offset + index] = leafValue;
        global_active_leaf_parent_leaf_index[newly_active_offset + index] = globalIndex;

        // If non-pendant paths were found, populate the search tree in the 
        // complete level
        for (int startingCLL = removeFromComplete; index < leavesFromCompleteLvl - removeFromComplete; ++index, ++startingCLL){
            //printf("global_newly_active_leaves[%d] = %d\n",newly_active_offset + index, leftMostLeafIndexOfFullLevel + index + removeFromComplete);
            global_newly_active_leaves[newly_active_offset + index] = leftMostLeafIndexOfFullLevel + startingCLL;
            global_active_leaf_parent_leaf_value[newly_active_offset + index] = leafValue;
            global_active_leaf_parent_leaf_index[newly_active_offset + index] = globalIndex;
            global_active_leaf_index[newly_active_offset + index] = newly_active_offset + index;
        }

        int totalNewActive = (leavesFromCompleteLvl - removeFromComplete) + leavesFromIncompleteLvl;
        printf("globalIndex %d, ParallelPopulateNewlyActivateLeafNodesBreadthFirstClean\n",globalIndex);
        printf("Leaves %d, completeLevel Level Depth %d\n",leavesToProcess, completeLevel);
        printf("Leaves %d, leavesFromCompleteLvl %d\n",leavesToProcess, leavesFromCompleteLvl);
        printf("Leaves %d, incompleteLevel Level Depth %d\n",leavesToProcess, incompleteLevel);
        printf("Leaves %d, treeSizeComplete %d\n",leavesToProcess, treeSizeComplete);
        printf("Leaves %d, totalNewActive %d\n",leavesToProcess, totalNewActive);

        // If non-pendant paths were found, populate the search tree in the 
        // incomplete level
        for (int incompleteIndex = 0; index < totalNewActive; ++index, ++incompleteIndex){
            //printf("global_newly_active_leaves[%d] = %d\n",newly_active_offset + index, leftMostLeafIndexOfIncompleteLevel + incompleteIndex);
            global_newly_active_leaves[newly_active_offset + index] = leftMostLeafIndexOfIncompleteLevel + incompleteIndex;
            global_active_leaf_parent_leaf_value[newly_active_offset + index] = leafValue;
            global_active_leaf_parent_leaf_index[newly_active_offset + index] = globalIndex;
            global_active_leaf_index[newly_active_offset + index] = newly_active_offset + index;
        }
        for (int testP = 0; testP < totalNewActive; ++testP){
            printf("leafValue %d new active %d new active's parent %d\n",leafValue, global_newly_active_leaves[newly_active_offset + testP],global_active_leaf_parent_leaf_value[newly_active_offset + testP]);
        }
    }
}

    /*
    int levelDepth = CalculateNumberOfFullLevels(leavesThatICanProcess);
    printf("Block ID %d thread  %d %s can process %d full levels\n", blockIdx.x, threadIdx.x, levelDepth);

    // int myLB = CalculateLevelOffset(levelDepth);
    // int myUB = CalculateLevelUpperBound(levelDepth);
    // int globalLevelOffset = CalculateLevelOffset(level + levelDepth);

    int lowestFullLevelSize = CalculateLevelSize(levelDepth);
    printf("Block ID %d thread  %d %s can process %d leaves in the lowers full levely\n", blockIdx.x, threadIdx.x, lowestFullLevelSize);

    int leftMostLeafIndexOfFullLevel = pow(3.0, levelDepth) * leafIndex;
    // [my LB, myUB] correspond to a subset of the level of the global tree
    //      0
    //    0 x 0
    // 000 yyy 000
    // For example consider vertex 'x'.
    // If it wanted to induce 1 level
    // [my LB, myUB] would correspond to the global leaf indices of the 'y's

    // This is for inducing the next full lowest level
    // I need to double check the math here.
    // for (int c = 1; c <= 3; ++c){
    //    graphs[3*leafIndex + c]
    /*
    int numberOfToSkipInFullLevel = CalculateNumberInIncompleteLevel(leavesThatICanProcess);

    // To skip activating a node in the full level with an active child
    // Ceiling Divide by 3
    int numberWithActiveChildren = (numberOfToSkipInFullLevel + 3 - 1) / 3;
    for (int child = numberWithActiveChildren + 1; child <= lowestFullLevelSize; ++child){
        global_active_vertices[leftMostLeafIndexOfFullLevel + child] = 1;
    }

    // Deactivates the members of the lowest full level
    // which have children lower than them
    int leftMostLeafIndexOfIncompleteLevel = pow(3.0, levelDepth+1) * leafIndex;
    for (int child = 1; child <= numberWithActiveChildren * 3; ++child){
        global_active_vertices[leftMostLeafIndexOfIncompleteLevel + child] = 1;
    }
  
}

  */
__global__ void ParallelProcessDegreeZeroVertices(
                            int numberOfRows,
                            int verticesRemainingInGraph,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_degrees_dev_ptr){


    int leafIndex = blockIdx.x;
    if (threadIdx.x == 0){
        printf("Leaf index %d Entered ProcessDeg0\n", leafIndex);
    }
    extern __shared__ int degreeZeroVertex[];

    int degreesOffset = leafIndex * numberOfRows;
    int remainingVerticesOffset = leafIndex * verticesRemainingInGraph;
    int numVertices = global_remaining_vertices_size_dev_ptr[leafIndex];
    int numVerticesRemoved = 0;
    int numIters = (numVertices + blockDim.x + 1 )/blockDim.x;
    //for (int iter = threadIdx.x; iter < numVertices; iter += blockDim.x){
    //    
    //}
    //__syncthreads();
    // Sync threads will hang for num verts > tPB...
    int iter = 0;
    int vertex;
    for (vertex = threadIdx.x; iter < numIters; ++iter, vertex += blockDim.x){
        // Prevent out of bound memory access by setting vertex to 0 for vertex > numVertices
        degreeZeroVertex[threadIdx.x] = (int)(0 == (global_degrees_dev_ptr[degreesOffset + global_remaining_vertices_dev_ptr[remainingVerticesOffset + vertex*((int)(vertex < numVertices))]]));
        // Set the garbage values to 0.  Since there is a sync threads in this for loop, we need
        // to round up the iters, and some threads won't correspond to actual remaining vertices.
        // Set these to 0.
        degreeZeroVertex[threadIdx.x] *= (int)(vertex < numVertices);

        // Makes this entry INT_MAX if degree 0
        // Leaves unaltered if not degree 0
        // Prevent out of bound memory access by setting vertex to 0 for vertex > numVertices
        global_remaining_vertices_dev_ptr[remainingVerticesOffset + vertex*((int)(vertex < numVertices))] += (INT_MAX - global_remaining_vertices_dev_ptr[remainingVerticesOffset + vertex*((int)(vertex < numVertices))])*degreeZeroVertex[threadIdx.x];        
        int i = blockDim.x/2;
        __syncthreads();
        while (i != 0) {
            if (threadIdx.x < i){
                //printf("degreeZeroVertex[%d] = %d + %d\n", threadIdx.x, degreeZeroVertex[threadIdx.x], degreeZeroVertex[threadIdx.x + i]);
                degreeZeroVertex[threadIdx.x] += degreeZeroVertex[threadIdx.x + i];
                degreeZeroVertex[threadIdx.x + i] = 0;
            }
            __syncthreads();
            i /= 2;
        }
        if (threadIdx.x == 0){
            numVerticesRemoved += degreeZeroVertex[threadIdx.x];
            printf("leafIndex %d numVerticesRemoved %d\n", leafIndex, numVerticesRemoved);
        }
    }
    // Update remaining vert size
    // Now just need to sort those INT_MAX entries to the end of the array
    if (threadIdx.x == 0){
        printf("leafIndex %d total numVerticesRemoved %d\n", leafIndex, numVerticesRemoved);
        global_remaining_vertices_size_dev_ptr[leafIndex] -= numVerticesRemoved;
    }
}

__global__ void ParallelProcessDegreeZeroVerticesClean(
                            int numberOfRows,
                            int verticesRemainingInGraph,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_edges_left_to_cover_count,
                            int * global_degrees_dev_ptr){


    int leafIndex = blockIdx.x;
    if (0 == global_remaining_vertices_size_dev_ptr[leafIndex])
        return;
    if (threadIdx.x == 0){
        printf("Leaf index %d Entered ProcessDeg0\n", leafIndex);
    }
    extern __shared__ int degreeZeroVertex[];

    int degreesOffset = leafIndex * numberOfRows;
    int remainingVerticesOffset = leafIndex * verticesRemainingInGraph;
    int numVertices = global_remaining_vertices_size_dev_ptr[leafIndex];
    int numVerticesRemoved = 0;
    int numIters = (numVertices + blockDim.x + 1 )/blockDim.x;
    //for (int iter = threadIdx.x; iter < numVertices; iter += blockDim.x){
    //    
    //}
    //__syncthreads();
    // Sync threads will hang for num verts > tPB...
    
    int vertex = threadIdx.x;
    int boundedVertex = threadIdx.x;
    int addend = 0;
    int vertexValue = 0;
    int degree = 0;
    for (int iter = 0; iter < numIters; ++iter, vertex += blockDim.x){
        boundedVertex = vertex*(int)(vertex < numVertices) + 0;
        vertexValue = global_remaining_vertices_dev_ptr[remainingVerticesOffset + boundedVertex];
        degree = global_degrees_dev_ptr[degreesOffset + vertexValue];
        // Prevent out of bound memory access by setting vertex to 0 for vertex > numVertices
        degreeZeroVertex[threadIdx.x] = (int)(0 == (degree));
        // Set the garbage values to 0.  Since there is a sync threads in this for loop, we need
        // to round up the iters, and some threads won't correspond to actual remaining vertices.
        // Set these to 0.
        degreeZeroVertex[threadIdx.x] *= (int)(vertex < numVertices);

        // Makes this entry INT_MAX if degree 0
        // Leaves unaltered if not degree 0
        // Prevent out of bound memory access by setting vertex to 0 for vertex > numVertices
        addend = 
        (INT_MAX - global_remaining_vertices_dev_ptr[remainingVerticesOffset 
        + boundedVertex])*degreeZeroVertex[threadIdx.x];

        global_remaining_vertices_dev_ptr[remainingVerticesOffset 
        + boundedVertex] += addend;        

        int i = blockDim.x/2;
        __syncthreads();
        while (i != 0) {
            if (threadIdx.x < i){
                //printf("degreeZeroVertex[%d] = %d + %d\n", threadIdx.x, degreeZeroVertex[threadIdx.x], degreeZeroVertex[threadIdx.x + i]);
                degreeZeroVertex[threadIdx.x] += degreeZeroVertex[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }
        if (threadIdx.x == 0){
            numVerticesRemoved += degreeZeroVertex[threadIdx.x];
            printf("leafIndex %d numVerticesRemoved %d\n", leafIndex, numVerticesRemoved);
        }
    }
    // Update remaining vert size
    // Now just need to sort those INT_MAX entries to the end of the array
    if (threadIdx.x == 0){
        printf("leafIndex %d total numVerticesRemoved %d\n", leafIndex, numVerticesRemoved);
        global_remaining_vertices_size_dev_ptr[leafIndex] -= numVerticesRemoved;
    }
}
/*
__global__ void ParallelActiveVertexPathOffsets(int * global_active_leaf_value,
                                                int global_active_leaf_indices_count,
                                                ){

    int leafIndex = levelOffset + blockIdx.x;
    if (leafIndex >= levelUpperBound)
        return;    

    printf("LevelAware RowOffs blockIdx %d is running\n", blockIdx.x);
    printf("LevelAware RowOffs leaf index %d is running\n", leafIndex);

    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int bufferRowOffsOffset = blockIdx.x * (numberOfRows + 1);

    for (int iter = threadIdx.x; iter < numberOfRows+1; iter += blockDim.x){
        global_cols_vals_segments[bufferRowOffsOffset + iter] = (blockIdx.x * numberOfEdgesPerGraph) + global_row_offsets_dev_ptr[rowOffsOffset + iter];
        printf("global_cols_vals_segments[bufferRowOffsOffset + %d] = %d + %d\n", iter, (blockIdx.x * numberOfEdgesPerGraph), global_row_offsets_dev_ptr[rowOffsOffset + iter]);

    }

    if(threadIdx.x == 0){
        printf("LevelAware RowOffs \n");
        for (int i = 0; i < numberOfRows+1; ++i){
            printf("global_cols_vals_segments[%d] = %d  \n",i, global_cols_vals_segments[bufferRowOffsOffset+i]);
        }
        printf("\n");
    }
}
*/

__global__ void ParallelCreateActiveVerticesRowOffsets(
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_cols_vals_segments,
                            int * global_set_inclusion_bool_ptr){

    int leafIndex = blockIdx.x;

    printf("LevelAware RowOffs blockIdx %d is running\n", blockIdx.x);
    printf("LevelAware RowOffs leaf index %d is running\n", leafIndex);

    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int bufferRowOffsOffset = blockIdx.x * (numberOfRows + 1);

    for (int iter = threadIdx.x; iter < numberOfRows+1; iter += blockDim.x){
        global_cols_vals_segments[bufferRowOffsOffset + iter] = (blockIdx.x * numberOfEdgesPerGraph) + global_row_offsets_dev_ptr[rowOffsOffset + iter];
        printf("global_cols_vals_segments[bufferRowOffsOffset + %d] = %d + %d\n", iter, (blockIdx.x * numberOfEdgesPerGraph), global_row_offsets_dev_ptr[rowOffsOffset + iter]);

    }
}

__global__ void SetVerticesRemaingSegements(int dLSPlus1,
                            int numberOfRows,
                            int * global_vertex_segments){
    for (int entry = threadIdx.x; entry < dLSPlus1; entry += blockDim.x){
        global_vertex_segments[entry] = entry * numberOfRows;
    }
}

__global__ void SetPathOffsets(int sDLSPlus1,
                               int * global_set_path_offsets){
    for (int entry = threadIdx.x; entry < sDLSPlus1; entry += blockDim.x){
        global_set_path_offsets[entry] = entry * threadsPerBlock;
    }
}
__device__ void SetOutgoingEdges(int rowOffsOffset,
                                int valsAndColsOffset,
                                int degreesOffset,
                                int u,
                                int * global_row_offsets_dev_ptr,
                                int * global_columns_dev_ptr,
                                int * global_values_dev_ptr,
                                int * global_degrees_dev_ptr){
    //int rowOffsOffset = leafIndex * (numberOfRows + 1);
    //int valsAndColsOffset = leafIndex * numberOfEdgesPerGraph;
    int uLB = global_row_offsets_dev_ptr[rowOffsOffset + u];
    int uUB = global_row_offsets_dev_ptr[rowOffsOffset + u + 1];    // Set out-edges
    for (int i = uLB; i < uUB; ++i){
        global_values_dev_ptr[valsAndColsOffset + i] = 0;
    }
    global_degrees_dev_ptr[degreesOffset + u] = 0;
}


__device__ void SetIncomingEdges(int rowOffsOffset,
                                int valsAndColsOffset,
                                int degreesOffset,
                                int u,
                                int * global_row_offsets_dev_ptr,
                                int * global_columns_dev_ptr,
                                int * global_values_dev_ptr,
                                int * global_degrees_dev_ptr){
    int v;
    int uLB = global_row_offsets_dev_ptr[rowOffsOffset + u];
    int uUB = global_row_offsets_dev_ptr[rowOffsOffset + u + 1];
    int vLB;
    int vUB;
        // Set out-edges
    for (int i = uLB; i < uUB; ++i){
        v = global_columns_dev_ptr[valsAndColsOffset + i];
        vLB = global_row_offsets_dev_ptr[rowOffsOffset + v];
        vUB = global_row_offsets_dev_ptr[rowOffsOffset + v + 1];
        for (int j = vLB; i < vUB; ++j){
            if(u == global_columns_dev_ptr[valsAndColsOffset + j]){
                global_values_dev_ptr[valsAndColsOffset + j] = 0;
                --global_degrees_dev_ptr[degreesOffset + v];
                break;
            }
        }
    }
}

void CallPopulateTree(int numberOfLevels, 
                    Graph & g){


    int maxDegree = g.GetLargestDegree();

    //int treeSize = 200000;
    int counters = 2;

    int expandedData = g.GetEdgesLeftToCover();
    int condensedData = g.GetVertexCount();
    numberOfLevels = g.GetVertexCount()/2 + 1;
    int secondDeepestLevelSize = CalculateDeepestLevelWidth(numberOfLevels-2);;
    int deepestLevelSize = CalculateDeepestLevelWidth(numberOfLevels-1);;
    long long bufferSize = deepestLevelSize;
    long long treeSize = CalculateSpaceForDesiredNumberOfLevels(numberOfLevels);

    int condensedData_plus1 = condensedData + 1;
    long long sizeOfSingleGraph = expandedData*2 + 2*condensedData + condensedData_plus1 + maxDegree + counters;
    long long totalMem = sizeOfSingleGraph * treeSize * sizeof(int) + 
        condensedData * bufferSize * sizeof(int) +
            2 * expandedData * bufferSize * sizeof(int);

    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        int id;
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        std::cout << "GPU " << id << " memory: free=" << free << ", total=" << total << std::endl;
    }

    std::cout << "You are about to allocate " << double(totalMem)/1024/1024/1024 << " GB" << std::endl;
    std::cout << "Your GPU RAM has " << double(free)/1024/1024/1024 << " GB available" << std::endl;
    /*
    do 
    {
        std::cout << '\n' << "Press enter to continue...; ctrl-c to terminate";
    } while (std::cin.get() != '\n');
    */
// Each of these will be wrapped in a cub double buffer
// which will switch the active set passed to the methods which
// rely on these pointers.  This way I only keep the relevant 
// data instead of the entire tree.
    int * global_row_offsets_dev_ptr;
    int * global_columns_dev_ptr;
    int * global_values_dev_ptr;
    int * global_degrees_dev_ptr; 
    int * global_remaining_vertices_dev_ptr;
    int * global_remaining_vertices_size_dev_ptr;

    int * global_row_offsets_dev_ptr_buffer;
    int * global_columns_dev_ptr_buffer;
    int * global_values_dev_ptr_buffer;
    int * global_degrees_dev_ptr_buffer; 
    int * global_remaining_vertices_dev_ptr_buffer;
    int * global_remaining_vertices_size_dev_ptr_buffer;


    int * global_paths_ptr;


    int * global_vertices_included_dev_ptr;
    int * global_pendant_path_bool_dev_ptr;
    int * global_pendant_path_reduced_bool_dev_ptr;
    int * global_pendant_child_dev_ptr;
    //
    int * global_paths_indices_ptr, * global_paths_indices_ptr_buffer; 
    int * global_set_inclusion_bool_ptr, * global_set_inclusion_bool_ptr_buffer;
    int * global_set_path_offsets;
    //
    int * global_reduced_set_inclusion_count_ptr;
    int * global_reduced_set_inclusion_count_ptr_buffer;

    int * global_edges_left_to_cover_count;
    int * global_edges_left_to_cover_count_buffer;

    int * global_active_leaf_value, * global_active_leaf_value_buffer;
    // Used to as the source of the copied graph
    int * global_active_leaf_parent_leaf_index, * global_active_leaf_parent_leaf_index_buffer;
    // Used as the destination for the graph
    int * global_active_leaf_index, * global_active_leaf_index_buffer;
    // Used for determining when to stop recursing up for set vertices
    int * global_active_leaf_parent_leaf_value, * global_active_leaf_parent_leaf_value_buffer;

    int * global_active_leaf_indices_count;
    int * global_active_leaf_indices_count_buffer;

    int * global_active_leaf_offset_ptr;
    int * global_active_leaf_offset_ptr_buffer;

    int * global_cols_vals_segments;
    int * global_vertex_segments;

    int max_dfs_depth = 4;
    int numberOfRows = g.GetNumberOfRows();
    int numberOfEdgesPerGraph = g.GetEdgesLeftToCover(); 
    int verticesRemainingInGraph = g.GetRemainingVertices().size(); 
    int activeLeavesPerNode = CalculateAmountOfMemoryToAllocatePerActiveLeaf();

    cudaMalloc( (void**)&global_row_offsets_dev_ptr, ((numberOfRows+1)*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_columns_dev_ptr, (numberOfEdgesPerGraph*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_values_dev_ptr, (numberOfEdgesPerGraph*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_degrees_dev_ptr, (numberOfRows*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_remaining_vertices_dev_ptr, (verticesRemainingInGraph*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_remaining_vertices_size_dev_ptr, deepestLevelSize * sizeof(int) );

    cudaMalloc( (void**)&global_row_offsets_dev_ptr_buffer, ((numberOfRows+1)*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_columns_dev_ptr_buffer, (numberOfEdgesPerGraph*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_values_dev_ptr_buffer, (numberOfEdgesPerGraph*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_degrees_dev_ptr_buffer, (numberOfRows*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_remaining_vertices_dev_ptr_buffer, (verticesRemainingInGraph*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_remaining_vertices_size_dev_ptr_buffer, deepestLevelSize * sizeof(int) );

    cub::DoubleBuffer<int> row_offsets(global_row_offsets_dev_ptr, global_row_offsets_dev_ptr_buffer);
    cub::DoubleBuffer<int> columns(global_columns_dev_ptr, global_columns_dev_ptr_buffer);
    cub::DoubleBuffer<int> values(global_values_dev_ptr, global_values_dev_ptr_buffer);
    cub::DoubleBuffer<int> degrees(global_degrees_dev_ptr, global_degrees_dev_ptr_buffer);
    cub::DoubleBuffer<int> remaining_vertices(global_remaining_vertices_dev_ptr, global_remaining_vertices_dev_ptr_buffer);
    cub::DoubleBuffer<int> remaining_vertices_count(global_remaining_vertices_size_dev_ptr, global_remaining_vertices_size_dev_ptr_buffer);


    cudaMalloc( (void**)&global_paths_ptr, (max_dfs_depth*secondDeepestLevelSize*threadsPerBlock) * sizeof(int) );
    

    cudaMalloc( (void**)&global_set_path_offsets, (secondDeepestLevelSize+1) * sizeof(int) );

    
    // Not global to the entire tree, overwritten every level

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    // Not global to the entire tree, overwritten every level
    // For sorting the threadsPerBlock paths by set inclusion in the MIS
    cudaMalloc( (void**)&global_paths_indices_ptr, secondDeepestLevelSize*threadsPerBlock*sizeof(int));
    cudaMalloc( (void**)&global_paths_indices_ptr_buffer, secondDeepestLevelSize*threadsPerBlock*sizeof(int));
    cub::DoubleBuffer<int> paths_indices(global_paths_indices_ptr, global_paths_indices_ptr_buffer);

    cudaMalloc( (void**)&global_set_inclusion_bool_ptr, threadsPerBlock * secondDeepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_set_inclusion_bool_ptr_buffer, threadsPerBlock * secondDeepestLevelSize * sizeof(int) );
    cub::DoubleBuffer<int> set_inclusion(global_set_inclusion_bool_ptr, global_set_inclusion_bool_ptr_buffer);

    cudaMalloc( (void**)&global_reduced_set_inclusion_count_ptr, deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_reduced_set_inclusion_count_ptr_buffer, deepestLevelSize * sizeof(int) );
    cub::DoubleBuffer<int> set_inclusion_count(global_reduced_set_inclusion_count_ptr, global_reduced_set_inclusion_count_ptr_buffer);

    //cudaMemset(global_remaining_vertices_dev_ptr, INT_MAX, (numberOfRows*deepestLevelSize) * sizeof(int));

    // Since we statically allocate vertices remaining and col/vals
    cudaMalloc( (void**)&global_vertex_segments, (deepestLevelSize+1) * sizeof(int) );
    cudaMalloc( (void**)&global_cols_vals_segments, (numberOfRows+1) * deepestLevelSize * sizeof(int) );


    cudaMalloc( (void**)&global_vertices_included_dev_ptr, 2 * treeSize * sizeof(int) );
    //cudaMemsetAsync(global_vertices_included_dev_ptr, 0, size_t(2 * treeSize) * sizeof(int));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_vertices_included_dev_ptr),  -1, size_t(2 * treeSize));
    // Hopefully using a negative int isn't a problem.
    //cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_vertices_included_dev_ptr),  -1, size_t(2 * treeSize));
    // Each node can process tPB pendant edges per call
    cudaMalloc( (void**)&global_pendant_path_bool_dev_ptr, threadsPerBlock * deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_pendant_path_reduced_bool_dev_ptr, deepestLevelSize * sizeof(int) );
    // Not global to the entire tree, overwritten every level
    cudaMalloc( (void**)&global_pendant_child_dev_ptr, threadsPerBlock * deepestLevelSize * sizeof(int) );

    cudaMalloc( (void**)&global_edges_left_to_cover_count, deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_edges_left_to_cover_count_buffer, deepestLevelSize * sizeof(int) );
    cub::DoubleBuffer<int> edges_left(global_edges_left_to_cover_count, global_edges_left_to_cover_count_buffer);

    // These two arrays direct the flow of the graph building process
    // If a vertex is not skipped and it has 0 paths produced, it needs to enter the DFS method
    // If a vertex is skipped, it doesn't do anything
    // If a vertex is not skipped and it has n > 0 paths produced, the largest number of
    // levels are induced, at least 1.
    
    cudaMalloc( (void**)&global_active_leaf_offset_ptr, (secondDeepestLevelSize+1) * sizeof(int) );
    cudaMalloc( (void**)&global_active_leaf_offset_ptr_buffer, (secondDeepestLevelSize+1) * sizeof(int) );
    cub::DoubleBuffer<int> active_leaf_offset(global_active_leaf_offset_ptr, global_active_leaf_offset_ptr_buffer);


    // For now I won't anticipate the decisions of deeper levels
    // Therefore, I can only work on a range
    // 0 < x < deepest level size vertices
    // x <= second deepest level size vertices
    // The minimum number of active leaves which produce a full tree is floor(log(threadsPerBlock)/log(3.0))
    // but we have to assume the worst, that is the entire second deepest level is full
    // and the last level is filled by inducing 3 children per leaf node

    // This would eliminate calculating the offsets of each active node into the buffer
    // With this much memory we can write in my section then sort globally decreasing.
    // I will do the serial calculation for a dramatic decrease in memory usage
    //cudaMalloc( (void**)&global_active_leaf_value, activeLeavesPerNode*secondDeepestLevelSize*sizeof(int) );
    //cudaMalloc( (void**)&global_active_leaf_value_buffer, activeLeavesPerNode*secondDeepestLevelSize*sizeof(int) );

    // If we want to use a compressed list creation of active vertices
    // We will precalculate where each active node will write in this compressed array
    cudaMalloc( (void**)&global_active_leaf_value, deepestLevelSize*sizeof(int) );
    cudaMalloc( (void**)&global_active_leaf_value_buffer, deepestLevelSize*sizeof(int) );
    cub::DoubleBuffer<int> active_leaves_value(global_active_leaf_value, global_active_leaf_value_buffer);

    // Since we skip internal nodes, each active leaf needs to know the parent index
    // from which cols, vals, remaining vertices are to be copied
    // The parent graphs are maintained in memory, until all the new leaves are processed
    // Once the memory of the new leaves excede available memory, the new leaves are
    // copied back to host, and the rest of the new leaves processed, only then 
    // can the old graph be replaced by a new graph and the process continued.
    cudaMalloc( (void**)&global_active_leaf_index, deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_active_leaf_index_buffer, deepestLevelSize * sizeof(int) );
    cub::DoubleBuffer<int> active_leaves_index(global_active_leaf_index, global_active_leaf_index_buffer);

    cudaMalloc( (void**)&global_active_leaf_indices_count, 1 * sizeof(int) );
    cudaMalloc( (void**)&global_active_leaf_indices_count_buffer, 1 * sizeof(int) );
    cub::DoubleBuffer<int> active_leaves_count(global_active_leaf_indices_count, global_active_leaf_indices_count_buffer);

    cudaMalloc( (void**)&global_active_leaf_parent_leaf_index, deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_active_leaf_parent_leaf_index_buffer, deepestLevelSize * sizeof(int) );
    cub::DoubleBuffer<int> parent_leaf_index(global_active_leaf_parent_leaf_index, global_active_leaf_parent_leaf_index_buffer);

    cudaMalloc( (void**)&global_active_leaf_parent_leaf_value, deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_active_leaf_parent_leaf_value_buffer, deepestLevelSize * sizeof(int) );
    cub::DoubleBuffer<int> parent_leaf_value(global_active_leaf_parent_leaf_value, global_active_leaf_parent_leaf_value_buffer);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    CopyGraphToDevice(g,
                    row_offsets.Current(),
                    columns.Current(),
                    values.Current(),
                    degrees.Current(),
                    numberOfEdgesPerGraph,
                    edges_left.Current(),
                    remaining_vertices.Current(),
                    remaining_vertices_count.Current(),
                    verticesRemainingInGraph,
                    active_leaves_value.Current(),
                    active_leaves_count.Current());

    int numberOfBlocksForOneThreadPerLeaf;
    numberOfLevels = g.GetVertexCount()/2 + 1;
    
    // For printing the result of exclusive prefix sum cub lib
    int * hostOffset = new int[deepestLevelSize+1];
    int * activeLeavesHostIndex = new int[deepestLevelSize+1];
    int * activeLeavesHostValue = new int[deepestLevelSize+1];
    int * activeParentHostIndex = new int[deepestLevelSize+1];
    int * activeParentHostValue = new int[deepestLevelSize+1];

    // For visualization
    int * active_leaves_host = new int[deepestLevelSize];
    int * active_parents_host = new int[deepestLevelSize];
    int * coverTree = new int[2 * treeSize]; // "zero-length" placeholder
    std::fill_n (coverTree, 2 * treeSize, -1);
    bool isDirected = false;
    int parent;
    int v1 = 0;
    int v2 = 0;
    int cycle = 0;
    std::string name = "main";
    std::string filename = "";
    DotWriter::RootGraph gVizWriter(isDirected, name);

    std::string subgraph1 = "activationOfLeaves";
    std::string subgraph2 = "searchTree";

    DotWriter::Subgraph * actLeaves = gVizWriter.AddSubgraph(subgraph1);
    DotWriter::Subgraph * searchTree = gVizWriter.AddSubgraph(subgraph2);
    std::map<std::string, DotWriter::Node *> nodeMapActLeaves;    
    std::map<std::string, DotWriter::Node *> nodeMapSearchTree;    

    // For visualization

    
    // One greater than secondDeepestLevelSize
    int dLSPlus1 = (deepestLevelSize + 1);
    int ceilOfDLSPlus1 = (dLSPlus1 + threadsPerBlock - 1) / threadsPerBlock;

    // It is imperative the CSR's don't shrink for these segments to remain valid
    // That means InduceSubgraph can't be used, we have to do memcpys
    // Create Segment Offsets for RemainingVertices once
    SetVerticesRemaingSegements<<<ceilOfDLSPlus1,threadsPerBlock>>>(dLSPlus1,
                                                    verticesRemainingInGraph,
                                                    global_vertex_segments);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    // One greater than secondDeepestLevelSize
    int sDLSPlus1 = (secondDeepestLevelSize + 1);
    int ceilOfSDLSPlus1 = (sDLSPlus1 + threadsPerBlock - 1) / threadsPerBlock;

    SetPathOffsets<<<ceilOfSDLSPlus1,threadsPerBlock>>>(sDLSPlus1,
                                                        global_set_path_offsets);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    bool notFirstCall = false;
    // When activeVerticesCount == 0, loop terminates
    int activeVerticesCount = 1;
    int oldActiveVerticesCount;
    int zero = 0;
    int maxActiveLeafVal = INT_MIN;
    int minParentLeafVal = INT_MAX;

    while(activeVerticesCount){
        if(notFirstCall){
            SetEdges<<<activeVerticesCount, threadsPerBlock>>>(
                                                            numberOfRows,
                                                            numberOfEdgesPerGraph,
                                                            active_leaves_value.Current(),
                                                            parent_leaf_value.Current(),
                                                            row_offsets.Current(),
                                                            columns.Current(),
                                                            values.Current(),
                                                            degrees.Current(),
                                                            edges_left.Current(),
                                                            remaining_vertices_count.Current(),
                                                            global_vertices_included_dev_ptr);


            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);                                                                 

            ParallelProcessDegreeZeroVerticesClean<<<activeVerticesCount,
                                    threadsPerBlock,
                                    threadsPerBlock*sizeof(int)>>>
                (numberOfRows,
                verticesRemainingInGraph,
                remaining_vertices.Current(),
                remaining_vertices_count.Current(),
                edges_left.Current(),
                degrees.Current());

            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);  

            ParallelRowOffsetsPrefixSumDevice<<<activeVerticesCount,threadsPerBlock>>>
                                               (numberOfEdgesPerGraph,
                                                numberOfRows,
                                                row_offsets.Current(),
                                                global_cols_vals_segments);

            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);                                                                 
                                                            
            RestoreDataStructuresAfterRemovingChildrenVertices( activeVerticesCount,
                                                                threadsPerBlock,
                                                                numberOfRows,
                                                                numberOfEdgesPerGraph,
                                                                verticesRemainingInGraph,
                                                                row_offsets,
                                                                columns,
                                                                values,
                                                                remaining_vertices,
                                                                global_cols_vals_segments,
                                                                global_vertex_segments);

        
            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);

            PrintData<<<1,1>>>(activeVerticesCount,
                numberOfRows,
                numberOfEdgesPerGraph, 
                verticesRemainingInGraph,
                row_offsets.Current(),
                columns.Current(),
                values.Current(),
                degrees.Current(),
                remaining_vertices.Current(),
                edges_left.Current(),
                remaining_vertices_count.Current());

            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);
                                                    
        }
        notFirstCall = true;        
        // 1 thread per leaf
        std::cout << "Calling DFS" << std::endl;
        // 1 block per leaf; tries tPB random paths in G
        // Hence threadsPerBlock*4,
        // Each thread checks it's path's pendant status
        // These booleans are reduced in shared memory
        // Hence + threadsPerBlock
        std::cout << "pendantNodeExists - true " << std::endl;

        // Assumes all edges are turned on.  We need to compress a graph
        // after processing the edges of pendant paths
        int sharedMemorySize = threadsPerBlock*4 + threadsPerBlock;
        ParallelDFSRandom<<<activeVerticesCount,threadsPerBlock,sharedMemorySize*sizeof(int)>>>
                            (numberOfRows,
                            numberOfEdgesPerGraph,
                            verticesRemainingInGraph,
                            active_leaves_value.Current(),
                            row_offsets.Current(),
                            columns.Current(),
                            remaining_vertices.Current(),
                            remaining_vertices_count.Current(),
                            degrees.Current(),
                            global_paths_ptr,
                            paths_indices.Current(),
                            global_pendant_path_bool_dev_ptr,
                            global_pendant_path_reduced_bool_dev_ptr,
                            global_pendant_child_dev_ptr,
                            edges_left.Current(),
                            remaining_vertices_count.Current());

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        SetPendantEdges<<<activeVerticesCount, threadsPerBlock>>>(
                                                            numberOfRows,
                                                            numberOfEdgesPerGraph,
                                                            row_offsets.Current(),
                                                            columns.Current(),
                                                            values.Current(),
                                                            degrees.Current(),
                                                            edges_left.Current(),
                                                            global_pendant_path_bool_dev_ptr,
                                                            global_pendant_child_dev_ptr,
                                                            remaining_vertices_count.Current());
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        // Everything seems to be working till this point, need to print inside this method.
        // The deg 0 vertices are in the middle of the remVerts list..
        std::cout << "Calling ParallelIdentifyVertexDisjointNonPendantPathsClean" << std::endl;
        ParallelIdentifyVertexDisjointNonPendantPathsClean<<<activeVerticesCount,
                                                        threadsPerBlock,
                                                        10*threadsPerBlock*sizeof(int)>>>
                                                        (numberOfRows,
                                                        numberOfEdgesPerGraph,
                                                        row_offsets.Current(),
                                                        columns.Current(),
                                                        values.Current(),
                                                        global_pendant_path_bool_dev_ptr,
                                                        global_pendant_child_dev_ptr,
                                                        global_paths_ptr,
                                                        set_inclusion.Current(),
                                                        set_inclusion_count.Current(),
                                                        edges_left.Current(),
                                                        remaining_vertices_count.Current());

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        std::cout << "Calling SortPathIndices" << std::endl;
        SortPathIndices(activeVerticesCount,
                        threadsPerBlock,
                        paths_indices,
                        set_inclusion,
                        global_set_path_offsets);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        std::cout << "Calling PrintSets" << std::endl;
        PrintSets<<<1,1>>>(activeVerticesCount,
                paths_indices.Current(),
                paths_indices.Alternate(),
                set_inclusion.Current(),
                set_inclusion.Alternate(),
                global_set_path_offsets);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        // 2*threadsPerBlock 
        // 0 to blockDim, the sorting index
        // blockDim to 5*blockDim, the path values
        std::cout << "Calling ParallelAssignMISToNodesBreadthFirstClean" << std::endl;
        // Need to test the recurrence relation.
        ParallelAssignMISToNodesBreadthFirstClean<<<activeVerticesCount,
                                               threadsPerBlock,
                                               (threadsPerBlock + threadsPerBlock*4)*sizeof(int)>>>(
                                        active_leaves_value.Current(),
                                        paths_indices.Current(),
                                        set_inclusion_count.Current(),
                                        global_paths_ptr,
                                        global_vertices_included_dev_ptr,
                                        edges_left.Current(),
                                        remaining_vertices_count.Current());

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        cudaMemcpy(active_leaves_count.Alternate(), &zero, 1*sizeof(int), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        numberOfBlocksForOneThreadPerLeaf = (activeVerticesCount + threadsPerBlock - 1) / threadsPerBlock;
        // Need to test the recurrence relation.
        ParallelCalculateOffsetsForNewlyActivateLeafNodesBreadthFirst<<<numberOfBlocksForOneThreadPerLeaf,threadsPerBlock,threadsPerBlock*sizeof(int)>>>(
                                        active_leaves_count.Current(),
                                        active_leaves_count.Alternate(),
                                        set_inclusion_count.Current(),
                                        active_leaf_offset.Current(),
                                        edges_left.Current(),
                                        remaining_vertices_count.Current());
        
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
        
        // Guarunteed to be both in bounds, since the maximum number of active vertices is deepest level
        // and active_leaf_offset is length deepest level + 1

        // This is necessary for the case where the number of active vertices decreased from
        // the previous iteration, then 1 item past activeVerticesCount may be non-zero
        // To hack the cubLibrary into turning counters -> offsets, one needs to allocate
        // num_items + 1, see : https://github.com/NVIDIA/cub/issues/367

        // Eventually we could zero out the active_leaf_offset.Alternate array nonsync,
        // which may be faster than stopping here and explcitly setting 1 past the end to zero
        cudaMemcpy((&(active_leaf_offset.Current())[activeVerticesCount]), &zero, 1*sizeof(int), cudaMemcpyHostToDevice);
        
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        CUBLibraryPrefixSumDevice(&activeVerticesCount,
                                  active_leaf_offset);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
        
        cudaMemcpy(&hostOffset[0], (int*)active_leaf_offset.Alternate(), (activeVerticesCount+1)*sizeof(int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        std::cout << "hostOffset" << std::endl;
        for (int i = 0; i < activeVerticesCount+1; ++i){
            std::cout << hostOffset[i] << " ";
        }
        std::cout << std::endl;


        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        // Just to test a single iteration
        printf("TRUE activeVerticesCount : %d\n", activeVerticesCount);
        // Need to test the recurrence relation.
        ParallelPopulateNewlyActivateLeafNodesBreadthFirstClean<<<numberOfBlocksForOneThreadPerLeaf,threadsPerBlock>>>(
                                        active_leaves_value.Current(),
                                        active_leaves_value.Alternate(),
                                        active_leaves_count.Current(),
                                        set_inclusion_count.Current(),
                                        active_leaf_offset.Alternate(),
                                        active_leaves_index.Alternate(),
                                        parent_leaf_index.Alternate(),
                                        parent_leaf_value.Alternate(),
                                        edges_left.Current(),
                                        remaining_vertices_count.Current());
        
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        oldActiveVerticesCount = activeVerticesCount;
        cudaMemcpy(&activeVerticesCount, active_leaves_count.Alternate(), 1*sizeof(int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        std::cout << "activeVerticesCount: " << activeVerticesCount << std::endl;
        cudaMemcpy(&activeLeavesHostIndex[0], (int*)active_leaves_index.Alternate(), (activeVerticesCount)*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&activeLeavesHostValue[0], (int*)active_leaves_value.Alternate(), (activeVerticesCount)*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&activeParentHostIndex[0], (int*)parent_leaf_index.Alternate(), (activeVerticesCount)*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&activeParentHostValue[0], (int*)parent_leaf_value.Alternate(), (activeVerticesCount)*sizeof(int), cudaMemcpyDeviceToHost);


        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);


        // I need the indices not the absolute vals here.
        std::cout << "activeLeavesHostIndices" << std::endl;
        for (int i = 0; i < activeVerticesCount; ++i){
            std::cout << activeLeavesHostIndex[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "activeLeavesHostValue" << std::endl;
        for (int i = 0; i < activeVerticesCount; ++i){
            std::cout << activeLeavesHostValue[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "activeParentHostIndex" << std::endl;
        for (int i = 0; i < activeVerticesCount; ++i){
            std::cout << activeParentHostIndex[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "activeParentHostValue" << std::endl;
        for (int i = 0; i < activeVerticesCount; ++i){
            std::cout << activeParentHostValue[i] << " ";
        }
        std::cout << std::endl;

        int * old_row_offs = row_offsets.Current();
        int * new_row_offs = row_offsets.Alternate();
        int * old_cols = columns.Current();
        int * new_cols = columns.Alternate();        
        int * old_vals = values.Current();
        int * new_vals = values.Alternate();
        int * old_degrees = degrees.Current();
        int * new_degrees = degrees.Alternate();
        int * old_verts_remain = remaining_vertices.Current();
        int * new_verts_remain = remaining_vertices.Alternate();
        int * old_edges_left = edges_left.Current();
        int * new_edges_left = edges_left.Alternate();
        int * old_verts_remain_count = remaining_vertices_count.Current();
        int * new_verts_remain_count = remaining_vertices_count.Alternate();

        int * old_active_leaves_count = active_leaves_count.Current();
        int * old_active_leaf_offset = active_leaf_offset.Current();
        int * old_active_leaves_index = active_leaves_index.Current();
        int * old_active_leaves_value = active_leaves_value.Current();
        int * old_parent_leaf_index = parent_leaf_index.Current();
        int * old_parent_leaf_value = parent_leaf_value.Current();

        // Memory-Unoptimized
        for(int newChild = 0; newChild < activeVerticesCount; ++newChild){
            cudaMemcpy(&new_row_offs[newChild*(numberOfRows+1)], &old_row_offs[activeParentHostIndex[newChild]*(numberOfRows+1)], (numberOfRows+1)*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&new_cols[newChild*numberOfEdgesPerGraph], &old_cols[activeParentHostIndex[newChild]*numberOfEdgesPerGraph], numberOfEdgesPerGraph*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&new_vals[newChild*numberOfEdgesPerGraph], &old_vals[activeParentHostIndex[newChild]*numberOfEdgesPerGraph], numberOfEdgesPerGraph*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&new_degrees[newChild*numberOfRows], &old_degrees[activeParentHostIndex[newChild]*numberOfRows], numberOfRows*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&new_verts_remain[newChild*verticesRemainingInGraph], &old_verts_remain[activeParentHostIndex[newChild]*verticesRemainingInGraph], verticesRemainingInGraph*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&new_edges_left[newChild], &old_edges_left[activeParentHostIndex[newChild]], 1*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&new_verts_remain_count[newChild], &old_verts_remain_count[activeParentHostIndex[newChild]], 1*sizeof(int), cudaMemcpyDeviceToDevice);      
        }

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        // Memory-Unoptimized
        const size_t row_offs_sz = size_t(oldActiveVerticesCount*(numberOfRows+1)) * sizeof(int);
        const size_t cols_vals_sz = size_t(oldActiveVerticesCount*numberOfEdgesPerGraph) * sizeof(int);
        const size_t degrees_sz = size_t(oldActiveVerticesCount*numberOfRows) * sizeof(int);
        const size_t verts_remain_sz = size_t(oldActiveVerticesCount*verticesRemainingInGraph) * sizeof(int);
        const size_t remain_count_edges_left_sz = size_t(oldActiveVerticesCount) * sizeof(int);
        const size_t active_vert_off_sz = size_t(oldActiveVerticesCount+1) * sizeof(int);
        const size_t active_leaves_count_sz = size_t(1) * sizeof(int);

        cudaMemsetAsync(old_row_offs, 0, row_offs_sz);
        cudaMemsetAsync(old_cols, 0, cols_vals_sz);
        cudaMemsetAsync(old_vals, 0, cols_vals_sz);
        cudaMemsetAsync(old_degrees, 0, degrees_sz);
        cudaMemsetAsync(old_verts_remain, 0, verts_remain_sz);
        cudaMemsetAsync(old_edges_left, 0, remain_count_edges_left_sz);
        cudaMemsetAsync(old_verts_remain_count, 0, remain_count_edges_left_sz);

        // Need to clean the offsets
        cudaMemsetAsync(old_active_leaves_count, 0, active_leaves_count_sz);
        cudaMemsetAsync(old_active_leaf_offset, 0, active_vert_off_sz);
        cudaMemsetAsync(old_active_leaves_index, 0, remain_count_edges_left_sz);
        cudaMemsetAsync(old_active_leaves_value, 0, remain_count_edges_left_sz);
        cudaMemsetAsync(old_parent_leaf_index, 0, remain_count_edges_left_sz);
        cudaMemsetAsync(old_parent_leaf_value, 0, remain_count_edges_left_sz);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        printf("Before flip active_leaves_value.selector %d\n", active_leaves_value.selector);

        // Flips Current and Alternate

        row_offsets.selector = !row_offsets.selector;
        columns.selector = !columns.selector;
        values.selector = !values.selector;
        degrees.selector = !degrees.selector;
        remaining_vertices.selector = !remaining_vertices.selector;
        edges_left.selector = !edges_left.selector;
        remaining_vertices_count.selector = !remaining_vertices_count.selector;

        active_leaf_offset.selector = !active_leaf_offset.selector;
        active_leaves_count.selector = !active_leaves_count.selector;
        active_leaf_offset.selector = !active_leaf_offset.selector;
        active_leaves_value.selector = !active_leaves_value.selector;
        active_leaves_index.selector = !active_leaves_index.selector;
        parent_leaf_index.selector = !parent_leaf_index.selector;
        parent_leaf_value.selector = !parent_leaf_value.selector;     

        printf("After flip active_leaves_value.selector %d\n", active_leaves_value.selector);


        PrintData<<<1,1>>>(activeVerticesCount,
                            numberOfRows,
                            numberOfEdgesPerGraph, 
                            verticesRemainingInGraph,
                            row_offsets.Current(),
                            columns.Current(),
                            values.Current(),
                            degrees.Current(),
                            remaining_vertices.Current(),
                            edges_left.Current(),
                            remaining_vertices_count.Current());

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        // This tells me how much of the cover tree I need to copy
        
        if (activeVerticesCount > 0){
            GetMaxLeafValue(activeVerticesCount+1,
                            active_leaves_value,
                            &maxActiveLeafVal);
            GetMinLeafValue(activeVerticesCount+1,
                parent_leaf_value,
                &minParentLeafVal);
        } else {
            maxActiveLeafVal = treeSize-1;
        }
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        cudaMemcpy(&coverTree[2*minParentLeafVal], &global_vertices_included_dev_ptr[2*minParentLeafVal], (2*maxActiveLeafVal + 2) * sizeof(int) , cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        // Since the graph doesnt grow uniformly, it is too difficult to only copy the new parts..
        for (int i = 0; i < activeVerticesCount; ++i){
            std::string node1Name = std::to_string(activeParentHostValue[i]);
            std::map<std::string, DotWriter::Node *>::const_iterator nodeIt1 = nodeMapActLeaves.find(node1Name);
            if(nodeIt1 == nodeMapActLeaves.end()) {
                nodeMapActLeaves[node1Name] = actLeaves->AddNode(std::to_string(activeParentHostValue[i]));
            }
            std::string node2Name = std::to_string(activeLeavesHostValue[i]);
            std::map<std::string, DotWriter::Node *>::const_iterator nodeIt2 = nodeMapActLeaves.find(node2Name);
            if(nodeIt2 == nodeMapActLeaves.end()) {
                nodeMapActLeaves[node2Name] = actLeaves->AddNode(std::to_string(activeLeavesHostValue[i]));
            }  
            actLeaves->AddEdge(nodeMapActLeaves[node1Name], nodeMapActLeaves[node2Name], nodeMapActLeaves[node1Name]->GetLabel() +" to "+ nodeMapActLeaves[node2Name]->GetLabel()); 
        }

        std::cout << "TREE" << std::endl;
        for (int i = minParentLeafVal; i <= maxActiveLeafVal; ++i){
            if (i == 0){
                std::cout << "root" << std::endl;
            } else {
                v1 = coverTree[i*2 - 1];
                v2 = coverTree[i*2];
                std::cout << "(" << v1 << ", " << v2 << ")";
            }
        }
        std::cout << std::endl;
        for (int i = minParentLeafVal; i <= maxActiveLeafVal; ++i){
            std::cout << "( " << i << " )";
        }
        std::cout << std::endl;
        

        for (int lowestLeaf = minParentLeafVal; lowestLeaf <= maxActiveLeafVal; ++lowestLeaf){
            if (lowestLeaf == 0){
                std::cout << "root" << std::endl;
                std::string node1Name = std::to_string(lowestLeaf);
                std::map<std::string, DotWriter::Node *>::const_iterator nodeIt1 = nodeMapSearchTree.find(node1Name);
                // New node
                if(nodeIt1 == nodeMapSearchTree.end()){
                    nodeMapSearchTree[node1Name] = searchTree->AddNode(std::to_string(lowestLeaf));
                }
            } else {
                v1 = coverTree[lowestLeaf*2 - 1];
                v2 = coverTree[lowestLeaf*2];
                if (v1 == -1 && v2 == -1){

                } else if (v1 != -1 && v2 != -1){
                    std::string node1Name = std::to_string(lowestLeaf);
                    std::map<std::string, DotWriter::Node *>::const_iterator nodeIt1 = nodeMapSearchTree.find(node1Name);
                    // New node
                    if(nodeIt1 == nodeMapSearchTree.end()) {
                        nodeMapSearchTree[node1Name] = searchTree->AddNode(std::to_string(lowestLeaf));
                        if (((lowestLeaf-1)/3) > 0)
                            parent = (lowestLeaf-1)/3;
                        else
                            parent = 0;
                        std::string node2Name = std::to_string(parent);
                        std::map<std::string, DotWriter::Node *>::const_iterator nodeIt2 = nodeMapSearchTree.find(node2Name);
                        if(nodeIt2 == nodeMapSearchTree.end()) {
                            std::cout << "Error in search tree creation! " << std::endl;
                            std::cout << "Cant create child before parent! " << std::endl;
                            exit(1);                    
                        } 
                        searchTree->AddEdge(nodeMapSearchTree[node1Name], nodeMapSearchTree[node2Name]); 
                    }
                } else {
                    std::cout << "Error in search tree creation! " << std::endl;
                    std::cout << "Should populate even and odd value together! " << std::endl;
                    exit(1);
                }
            }
        }

        // Should always overwrite the whole tree.  Simple but slow
        filename = "Active_leaves_cycle_" + std::to_string(cycle) + ".dot";
        ++cycle;
        gVizWriter.WriteToFile(filename);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        std::cout << "You are about to start another loop " << std::endl;
        /*
        do 
        {
            std::cout << '\n' << "Press enter to continue...; ctrl-c to terminate";
        } while (std::cin.get() != '\n');
        */
    }


    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    cudaFree( global_row_offsets_dev_ptr );
    cudaFree( global_columns_dev_ptr );
    cudaFree( global_values_dev_ptr );
    cudaFree( global_degrees_dev_ptr );
    cudaFree( global_paths_ptr );
    cudaFree( global_edges_left_to_cover_count );
    cudaDeviceSynchronize();
}

void CopyGraphToDevice( Graph & g,
                        int * global_row_offsets_dev_ptr,
                        int * global_columns_dev_ptr,
                        int * global_values_dev_ptr,
                        int * global_degrees_dev_ptr,
                        int numberOfEdgesPerGraph,
                        int * global_edges_left_to_cover_count,
                        int * global_remaining_vertices_dev_ptr,
                        int * global_remaining_vertices_size_dev_ptr,
                        int verticesRemainingInGraph,
                        int * global_active_leaf_value,
                        int * global_active_leaf_indices_count){

    int * new_degrees_ptr = thrust::raw_pointer_cast(g.GetNewDegRef().data());
    int * vertices_remaining_ptr = thrust::raw_pointer_cast(g.GetRemainingVertices().data());
    
    std::cout << "remaining verts" << std::endl;
    for (auto & v : g.GetRemainingVertices())
        std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "remaining verts size " << g.GetRemainingVertices().size() << std::endl;

    // Graph vectors
    cudaMemcpy(global_degrees_dev_ptr, new_degrees_ptr, g.GetNumberOfRows() * sizeof(int),
                cudaMemcpyHostToDevice);
    cudaMemcpy(global_edges_left_to_cover_count, &numberOfEdgesPerGraph, 1 * sizeof(int),
                cudaMemcpyHostToDevice);
    cudaMemcpy(global_remaining_vertices_dev_ptr, vertices_remaining_ptr, g.GetRemainingVertices().size() * sizeof(int),
            cudaMemcpyHostToDevice);         
    cudaMemcpy(global_remaining_vertices_size_dev_ptr, &verticesRemainingInGraph, 1 * sizeof(int),
            cudaMemcpyHostToDevice);    
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
    // CSR vectors
    thrust::device_vector<int> old_row_offsets_dev = *(g.GetCSR().GetOldRowOffRef());
    thrust::device_vector<int> old_column_indices_dev = *(g.GetCSR().GetOldColRef());

    // SparseMatrix vectors
    thrust::device_vector<int> new_values_dev = g.GetCSR().GetNewValRef();
    thrust::device_vector<int> remaining_vertices_dev = g.GetRemainingVertices();
    // CSR pointers
    int * old_row_offsets_dev_ptr = thrust::raw_pointer_cast(old_row_offsets_dev.data());
    int * old_column_indices_dev_ptr = thrust::raw_pointer_cast(old_column_indices_dev.data());
    
    // SparseMatrix pointers
    int * new_values_dev_ptr = thrust::raw_pointer_cast(new_values_dev.data());

    // Currenly only sets the first graph in the cuda memory
    // Might as well be host code
    CalculateNewRowOffsets<<<1,1>>>(g.GetNumberOfRows(),
                                        0,
                                        1,
                                        global_row_offsets_dev_ptr,
                                        global_degrees_dev_ptr); 

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
    // Currenly only sets the first graph in the cuda memory
    InduceSubgraph<<<1,threadsPerBlock>>>(g.GetNumberOfRows(), 
                            g.GetEdgesLeftToCover(),          
                            old_row_offsets_dev_ptr,
                            old_column_indices_dev_ptr,
                            new_values_dev_ptr,
                            global_row_offsets_dev_ptr,
                            global_columns_dev_ptr,
                            global_values_dev_ptr);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    std::cout << "Activate root of tree" << std::endl;
    // Eventually replace zero with the variable starting index of the new tree
    int zero = 0;
    int one = 1;
    cudaMemcpy(global_active_leaf_value, &zero, 1*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(global_active_leaf_indices_count, &one, 1*sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "Activated root of tree" << std::endl;

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    thrust::device_ptr<int> back2Host_ptr = thrust::device_pointer_cast(global_columns_dev_ptr);
    thrust::device_vector<int> back2Host(back2Host_ptr, back2Host_ptr + g.GetEdgesLeftToCover());
    
    thrust::host_vector<int> hostFinal = back2Host;
    std::cout << "Priting data copied there and back" << std::endl;
    std::cout << "Size" << g.GetEdgesLeftToCover() << std::endl;
    for (auto & v : hostFinal)
        std::cout << v << " ";
    std::cout << std::endl;
}

#endif
