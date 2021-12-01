#ifdef FPT_CUDA

#include "ParallelB1_GPU.cuh"
#include <math.h>       /* pow */
#include "cub/cub.cuh"
#include "Random123/boxmuller.hpp"

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
                                    int * global_verts_tree,
                                    int * global_vertex_buffer,
                                    int * printAlt,
                                    int * printCurr){
    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;
    printf("Tree\n");
    for (int g = 0; g < (activeVerticesCount); ++g){
        printf("\n");
        for (int i = 0; i < numberOfRows; ++i){
            printf("%d ",global_verts_tree[g*numberOfRows + i]);
        }
    }
    printf("\n");

    printf("Buffer\n");
    for (int g = 0; g < (activeVerticesCount); ++g){
        printf("\n");
        for (int i = 0; i < numberOfRows; ++i){
            printf("%d ",global_vertex_buffer[g*numberOfRows + i]);
        }
    }
    printf("\n");

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
        printf("Index : %d\n", v);
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
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, paths_indices, set_inclusion,
        num_items, num_segments, global_set_path_offsets, global_set_path_offsets + 1);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, paths_indices, set_inclusion,
        num_items, num_segments, global_set_path_offsets, global_set_path_offsets + 1);

    cudaFree(d_temp_storage);

}

/*
__host__ void RestoreDataStructuresAfterRemovingChildrenVertices(int activeVerticesCount,
                                                                            int threadsPerBlock,
                                                                            int numberOfRows,
                                                                            int numberOfEdgesPerGraph,
                                                                            int * global_row_offsets_dev_ptr,
                                                                            int * global_offsets_buffer,
                                                                            int * global_column_buffer,
                                                                            int * global_value_buffer,
                                                                            int * global_vertex_buffer,
                                                                            int * global_vertex_segments,
                                                                            int * global_remaining_vertices_dev_ptr,
                                                                            int * global_columns_dev_ptr,
                                                                            int * global_values_dev_ptr){
    // Create pointer that starts at beginning of level
    // Leaves are indexed from 0; so I need to add the offset
    // of the leaf from the left of the tree * (numberOfRows+1) so the 
    // sorting operation works on an entire level.
    // global_offsets_buffer = &global_row_offsets_dev_ptr[levelOffset*(numberOfRows+1)];
    ParallelCreateLevelAwareRowOffsets<<<activeVerticesCount,threadsPerBlock>>>
                                        (activeVerticesCount,
                                        numberOfRows,
                                        numberOfEdgesPerGraph,
                                        global_row_offsets_dev_ptr,
                                        global_offsets_buffer);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    // Determine temporary device storage requirements
    int     *global_vertices_tree = NULL;
    // Determine temporary device storage requirements
    int     *global_columns_tree = NULL;
    // Determine temporary device storage requirements
    int     *global_values_tree = NULL;

    global_columns_tree = &global_columns_dev_ptr[levelOffset*numberOfEdgesPerGraph];
    global_values_tree = &global_values_dev_ptr[levelOffset*numberOfEdgesPerGraph];

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int> d_values(global_columns_tree, global_column_buffer);
    cub::DoubleBuffer<int> d_keys(global_values_tree, global_value_buffer);

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    int num_items = (activeVerticesCount)*numberOfEdgesPerGraph;
    int num_segments = (activeVerticesCount)*(numberOfRows+1);

    // Since vertices in a level follow each other, we reuse gob iterator
    // When we have active vertices from different levels, we will need 2 iterators
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values,
        num_items, num_segments, global_offsets_buffer, global_offsets_buffer + 1);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values,
        num_items, num_segments, global_offsets_buffer, global_offsets_buffer + 1);

    cudaFree(d_temp_storage);

    int * printAlt = d_keys.Alternate();
    int * printCurr = d_keys.Current();

    PrintEdges<<<1,1>>>  (levelOffset,
                levelUpperBound,
                numberOfRows,
                numberOfEdgesPerGraph,
                global_values_tree,
                global_value_buffer,
                printAlt,
                printCurr);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
                // Determine temporary device storage requirements
    void     *d_temp_storage2 = NULL;
    temp_storage_bytes = 0;
    num_items = (activeVerticesCount)*numberOfRows;
    num_segments = activeVerticesCount;

    global_vertices_tree = &global_remaining_vertices_dev_ptr[levelOffset*numberOfRows];
    cub::DoubleBuffer<int> d_keys_verts(global_vertices_tree, global_vertex_buffer);

    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage2, temp_storage_bytes, d_keys_verts,
        num_items, num_segments, global_vertex_segments, global_vertex_segments + 1);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage2, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage2, temp_storage_bytes, d_keys_verts,
        num_items, num_segments, global_vertex_segments, global_vertex_segments + 1);

    cudaFree(d_temp_storage2);

    printAlt = d_keys_verts.Alternate();
    printCurr = d_keys_verts.Current();

    PrintVerts<<<1,1>>>  (levelOffset,
                levelUpperBound,
                numberOfRows,
                global_vertices_tree,
                global_vertex_buffer,
                printAlt,
                printCurr);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
}
*/
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

__global__ void SetEdges(int numberOfRows,
                        int numberOfEdgesPerGraph,
                        int levelOffset,
                        int levelUpperBound,
                        int * global_row_offsets_dev_ptr,
                        int * global_columns_dev_ptr,
                        int * global_values_dev_ptr,
                        int * global_paths_ptr,
                        int * global_paths_length,
                        int * global_edges_left_to_cover_count){

    int leafIndex = levelOffset + blockIdx.x;
    if (leafIndex >= levelUpperBound) return;

    int threadIndex = threadIdx.x;

    int rowOffsOffset = (numberOfRows + 1) * (leafIndex-1)/3;
    int valsAndColsOffset = numberOfEdgesPerGraph * leafIndex;
    if (global_row_offsets_dev_ptr[rowOffsOffset + numberOfRows -1] != global_edges_left_to_cover_count[(leafIndex-1)/3])
        printf("\n\n\nERROR in the row offsets (%d) and number of columns (%d)\n\n\n", 
            global_row_offsets_dev_ptr[rowOffsOffset + numberOfRows -1],
            global_edges_left_to_cover_count[(leafIndex-1)/3]);
    int children[2], LB, UB, v, vLB, vUB;
    // Parent's DFS path
    int pathsOffset = ((leafIndex-1)/3) * 4;
/*
child x (path[0]) (path[2]);

        (path[1]) (path[3]);     

child y         or

        (path[1]) (path[0]);    

child z (path[2]) (path[1]);

Can't figure out a way to avoid these if conditionals without a kernel call to classify before this kernel is called.
*/
    int pathType = leafIndex % 3;
    if (pathType == 0){
        children[0] = global_paths_ptr[pathsOffset];
        children[1] = global_paths_ptr[pathsOffset + 2];
    } else if (pathType == 1) { 
        children[0] = global_paths_ptr[pathsOffset + 1];
        children[1] = global_paths_ptr[pathsOffset + 2];
    } else {
        children[0] = global_paths_ptr[pathsOffset + 1];
        if (global_paths_ptr[pathsOffset] == global_paths_ptr[pathsOffset + 3])
            children[1] = global_paths_ptr[pathsOffset];
        else
            children[1] = global_paths_ptr[pathsOffset + 3];
    }
    // Set out-edges
    for (int i = 0; i < 2; ++i){
        LB = global_row_offsets_dev_ptr[rowOffsOffset + children[i]];
        UB = global_row_offsets_dev_ptr[rowOffsOffset + children[i] + 1];    
        for (int edge = LB + threadIndex; edge < UB; edge += blockDim.x){
            global_values_dev_ptr[valsAndColsOffset + edge] = 0;
        }
    }
    __syncthreads();
    if (threadIndex == 0 && blockIdx.x == 0){
        printf("Block %d, levelOffset %d, leafIndex %d, children removed %d %d\n", blockIdx.x, levelOffset, leafIndex, children[0], children[1]);
        for (int i = 0; i < global_edges_left_to_cover_count[(leafIndex-1)/3]; ++i){
            printf("(%d, %d) ",global_columns_dev_ptr[valsAndColsOffset + i], global_values_dev_ptr[valsAndColsOffset + i]);
        }
        printf("\n");
    }
    // (u,v) is the form of edge pairs.  We are traversing over v's outgoing edges, 
    // looking for u as the destination and turning off that edge.
    // this may be more elegantly handled by 
    // (1) an associative data structure
    // (2) an undirected graph 
    // Parallel implementations of both of these need to be investigated.
    for (int i = 0; i < 2; ++i){
        LB = global_row_offsets_dev_ptr[rowOffsOffset + children[i]];
        UB = global_row_offsets_dev_ptr[rowOffsOffset + children[i] + 1];    // Set out-edges
        for (int edge = LB + threadIndex; edge < UB; edge += blockDim.x){
            v = global_columns_dev_ptr[valsAndColsOffset + edge];
            // guarunteed to only have one incoming and one outgoing edge connecting (x,y)
            vLB = global_row_offsets_dev_ptr[rowOffsOffset + v];
            vUB = global_row_offsets_dev_ptr[rowOffsOffset + v + 1];
            for (int outgoingEdgeOfV = vLB + threadIndex; 
                    outgoingEdgeOfV < vUB; 
                        outgoingEdgeOfV += blockDim.x){
                if (children[i] == global_columns_dev_ptr[valsAndColsOffset + outgoingEdgeOfV]){
                    // Set in-edge
                    global_values_dev_ptr[valsAndColsOffset + outgoingEdgeOfV] = 0;
                }
            }
        }
    }
    __syncthreads();
    if (threadIndex == 0 && blockIdx.x == 0){
        printf("Block %d, levelOffset %d, leafIndex %d, children removed %d %d\n", blockIdx.x, levelOffset, leafIndex, children[0], children[1]);
        for (int i = 0; i < global_edges_left_to_cover_count[(leafIndex-1)/3]; ++i){
            printf("(%d, %d) ",global_columns_dev_ptr[valsAndColsOffset + i], global_values_dev_ptr[valsAndColsOffset + i]);
        }
        printf("\n");
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
                                ) {
/*
        InduceSubgraph(
        numberOfRows,
        old_row_offsets_dev,
        old_columns_dev,
        old_values_dev,
        global_row_offsets_dev_ptr,
        global_columns_dev_ptr); */

     return;
}

// Single thread per leaf
__global__ void CreateSubsetOfRemainingVerticesLevelWise(int levelOffset,
                                                int levelUpperBound,
                                                int numberOfRows,
                                                int * global_degrees_dev_ptr,
                                                int * global_vertices_remaining,
                                                int * global_vertices_remaining_count){
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int leafIndex = levelOffset + threadID;
    if (leafIndex >= levelUpperBound) return;
    int degreesOffset = leafIndex * numberOfRows;

    global_vertices_remaining_count[leafIndex] = 0;

    for (int i = 0; i < numberOfRows; ++i){
        printf("Thread %d, global_degrees_dev_ptr[degreesOffset+%d] : %d\n", threadID, i, global_degrees_dev_ptr[degreesOffset+i]);
        if (global_degrees_dev_ptr[degreesOffset+i] == 0){
            continue;
        } else {
            global_vertices_remaining[degreesOffset+global_vertices_remaining_count[leafIndex]] = i;
            printf("Thread %d, global_vertices_remaining[degreesOffset+%d] : %d\n", threadID, global_vertices_remaining_count[leafIndex], global_vertices_remaining[degreesOffset+global_vertices_remaining_count[leafIndex]]);
            ++global_vertices_remaining_count[leafIndex];
        }
    }
}

// Single thread per leaf
/*
__global__ void DFSLevelWise(
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
                            int * global_outgoing_edge_vertices_count){
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int leafIndex = levelOffset + threadID;
    if (leafIndex >= levelUpperBound) return;
    int degreesOffset = leafIndex * numberOfRows;
    int pathsOffset = leafIndex * 4;
    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int valsAndColsOffset = leafIndex * numberOfEdgesPerGraph;
    int outgoingEdgeOffset = leafIndex * maxDegree;

    unsigned int counter = 0;
    ulong seed = 0;
    int randomNumber = randomGPU(counter, leafIndex, seed);
    int randomIndex = randomNumber % global_vertices_remaining_count[leafIndex];
    int randomVertex = global_vertices_remaining[degreesOffset+randomIndex];
    global_paths_ptr[pathsOffset + 0] = randomVertex;
    global_paths_length[leafIndex]++;
    printf("Thread %d, randomVertex : %d, path position : %d\n\n", threadID, randomVertex, 0);
// dfs 
    for (int i = 1; i < 4; ++i){
        global_outgoing_edge_vertices_count[leafIndex] = 0;
        for (int j = global_row_offsets_dev_ptr[rowOffsOffset + randomVertex]; 
                j < global_row_offsets_dev_ptr[rowOffsOffset + randomVertex + 1]; ++j){
            printf("Thread %d, global_values_dev_ptr[valsAndColsOffset + %d] : %d\n", threadID, j, global_values_dev_ptr[valsAndColsOffset + j]);
            if (global_values_dev_ptr[valsAndColsOffset + j] == 0)
                continue;
            else {
                global_outgoing_edge_vertices[outgoingEdgeOffset+global_outgoing_edge_vertices_count[leafIndex]] = global_columns_dev_ptr[valsAndColsOffset + j];
                printf("Thread %d, global_outgoing_edge_vertices[outgoingEdgeOffset+%d] : %d\n", threadID, global_outgoing_edge_vertices_count[leafIndex], global_outgoing_edge_vertices[outgoingEdgeOffset+global_outgoing_edge_vertices_count[leafIndex]]);
                ++global_outgoing_edge_vertices_count[leafIndex];
            }
        }
        ++counter;
        randomNumber = randomGPU(counter, leafIndex, seed);
        randomIndex = randomNumber % global_outgoing_edge_vertices_count[leafIndex];
        randomVertex = global_columns_dev_ptr[valsAndColsOffset + global_outgoing_edge_vertices[outgoingEdgeOffset+randomIndex]];
        
        if (randomVertex == global_paths_ptr[pathsOffset + i - 1]){
            // if degree is greater than 1 there exists an alternative path 
            // which doesnt form a simple cycle
            if (global_degrees_dev_ptr[degreesOffset+randomVertex] > 1){
                // Non-deterministic time until suitable edge which 
                // doesn't form a simple cycle is found.
                while(randomVertex == global_paths_ptr[pathsOffset + i - 1]){
                    ++counter;
                    randomNumber = randomGPU(counter, leafIndex, seed);
                    randomIndex = randomNumber % global_outgoing_edge_vertices_count[leafIndex];
                    randomVertex = global_columns_dev_ptr[valsAndColsOffset + global_outgoing_edge_vertices[outgoingEdgeOffset+randomIndex]];
                }
            } else {
                break;
            }
        }
        global_paths_ptr[pathsOffset + i] = randomVertex;
        ++global_paths_length[leafIndex];
        printf("Thread %d, randomVertex : %d, path position : %d\n\n", threadID, randomVertex, i);
        printf("Thread %d, global_paths_ptr[pathsOffset + %d] : %d", threadID, i, global_paths_ptr[pathsOffset + i]);
    }
}
*/
__global__ void GetRandomVertex(int levelOffset,
                                int levelUpperBound,
                                int numberOfRows,
                                int * global_remaining_vertices_dev_ptr,
                                int * global_remaining_vertices_size_dev_ptr,
                                int * global_paths_ptr){

    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    // Since each thread calculates four random numbers
    int leafIndex = levelOffset + threadID * 4;
    if (leafIndex >= levelUpperBound) return;

    RNG::ctr_type r;
    unsigned int counter = 0;
    ulong seed = 0;
    int remainingVertsOffset, pathsOffset, iteration, remainingVerticesSize; 
    remainingVertsOffset = leafIndex * numberOfRows;
    pathsOffset = leafIndex * 4;
    // r contains 4 random ints
    r = randomGPU_four(counter, leafIndex, seed);
    for(iteration = 0; iteration < 4 && (leafIndex + iteration) < levelUpperBound; ++iteration){
        remainingVerticesSize = global_remaining_vertices_size_dev_ptr[leafIndex];
        global_paths_ptr[pathsOffset] = r[iteration] % remainingVerticesSize;
        printf("Thread %d, leafIndex %d, random vertex %d", threadID, leafIndex, global_paths_ptr[pathsOffset]);
        remainingVertsOffset += numberOfRows;
        pathsOffset += 4;
    }
}

__global__ void GetRandomVertexSharedMem(int levelOffset,
                                int levelUpperBound,
                                int numberOfRows,
                                int * global_remaining_vertices_dev_ptr,
                                int * global_remaining_vertices_size_dev_ptr,
                                int * global_paths_ptr){

    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    // Since each thread calculates four random numbers
    int leafIndex = levelOffset + threadID * 4;
    extern __shared__ RNG::ctr_type random123Objects[];
    unsigned int counter = 0;
    ulong seed = 0;
    int r123Index = threadIdx.x / 4;
    
    if (threadIdx.x % 4 == 0)
       random123Objects[r123Index] = randomGPU_four(counter, leafIndex, seed);
    __syncthreads();

    if(leafIndex < levelUpperBound){
        int remainingVertsOffset = leafIndex * numberOfRows;
        int pathsOffset = leafIndex * 4;
        int randomNumIndex = threadIdx.x % 4;
        int remainingVerticesSize = global_remaining_vertices_size_dev_ptr[remainingVertsOffset];
        global_paths_ptr[pathsOffset] = (random123Objects[r123Index])[randomNumIndex] % remainingVerticesSize;
    }
}

// It is very important to only use the value in global_pendant_child_dev_ptr[]
// if global_pendant_path_bool_dev_ptr[] is true.  Otherwhise this path shouldnt
// be processed in the next step.

__global__ void ParallelDFSRandom(
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_active_leaf_indices,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_degrees_dev_ptr,
                            int * global_paths_ptr,
                            int * global_paths_indices_ptr,
                            int * global_pendant_path_bool_dev_ptr,
                            int * global_pendant_path_reduced_bool_dev_ptr,
                            int * global_pendant_child_dev_ptr){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Entered DFS\n");
        printf("\n");
    }
    int leafIndex = blockIdx.x;
    int leafValue = global_active_leaf_indices[leafIndex];

    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Set leafIndex\n");
        printf("\n");
    }
    // Initialized to 0, so will always perform DFS on first call
    // Subsequently, only perform DFS on pendant edges, so nonpendant false
    //if (global_pendant_path_bool_dev_ptr[leafIndex + threadIdx.x])
    //    return;
    int globalPathOffset = leafIndex * 4 * blockDim.x;
    int sharedMemPathOffset = threadIdx.x * 4;
    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int valsAndColsOffset = leafIndex * numberOfEdgesPerGraph;
    int degreesOffset = leafIndex * numberOfRows;
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
    int remainingVerticesSize = global_remaining_vertices_size_dev_ptr[leafIndex];
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("remainingVerticesSize %d\n", remainingVerticesSize);
        printf("\n");
    }
    int outEdgesCount;
    r = randomGPU_four(counter, leafValue, seed);
    // Random starting point
    pathsAndPendantStatus[sharedMemPathOffset + iteration] = global_remaining_vertices_dev_ptr[degreesOffset + (r[iteration] % remainingVerticesSize)];
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("pathsAndPendantStatus %d\n", pathsAndPendantStatus[sharedMemPathOffset + iteration]);
        printf("\n");
    }
    ++iteration;

    // Set random out at depth 1
    int randomVertRowOff = global_row_offsets_dev_ptr[rowOffsOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]];
    // Using degrees allow us to ignore the edges which have been turned off
        printf("randomVertRowOff %d\n", randomVertRowOff);
        printf("degreesOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1] %d\n",degreesOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]);
        printf("degreesOffset  %d\n",degreesOffset);
        printf("pathsAndPendantStatus[sharedMemPathOffset + iteration - 1] %d\n",pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]);
        outEdgesCount = global_degrees_dev_ptr[degreesOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]];
        printf("outEdgesCount %d\n", outEdgesCount);    
    //outEdgesCount = global_row_offsets_dev_ptr[rowOffsOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1] + 1]
    //                - randomVertRowOff;
        printf("valsAndColsOffset + randomVertRowOff + (r[iteration] mod outEdgesCount) %d\n", valsAndColsOffset + randomVertRowOff + (r[iteration] % outEdgesCount));
    
    // Assumes the starting point isn't degree 0
    pathsAndPendantStatus[sharedMemPathOffset + iteration] =  global_columns_dev_ptr[valsAndColsOffset + randomVertRowOff + (r[iteration] % outEdgesCount)];
    ++iteration;
        printf("(r[iteration] mod outEdgesCount) %d\n", (r[iteration] % outEdgesCount));

    if (threadIdx.x == 0){
        printf("Block %d, leafValue %d, got through first 2 iterations\n", blockIdx.x, leafValue);
        printf("\n");
    }
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
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Block %d, leafValue %d, got through last 2 iterations\n", blockIdx.x, leafValue);
        printf("\n");
    }
    printf("Thread %d (path %d -> %d -> %d -> %d) is %s\n", threadIdx.x, 
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
    int childIndex = pathsAndPendantStatus[sharedMemPathOffset + 0] != pathsAndPendantStatus[sharedMemPathOffset + 2];

    // or
    // Case 2 - length 3
    // v, v1, v2
    // if path[0] != path[2] was true, then path[1] == path[3]
    // cI == 1, since true casted to int is 1
    // Desired child is v1
    // Therefore, v1 == path[cI]
    global_pendant_child_dev_ptr[blockIdx.x*blockDim.x + threadIdx.x] = pathsAndPendantStatus[sharedMemPathOffset + childIndex];
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

__global__ void ParallelProcessPendantEdges(
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_active_leaf_indices,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_degrees_dev_ptr,
                            int * global_pendant_path_bool_dev_ptr,
                            int * global_pendant_child_dev_ptr){

    if (threadIdx.x == 0){
        printf("Block ID %d Started ParallelProcessPendantEdges\n", blockIdx.x);
        printf("\n");
    }
    // Only process pendant edges
    // 1 block per child, up to 32 children per node
    if (!global_pendant_path_bool_dev_ptr[blockIdx.x])
        return;
    if (threadIdx.x == 0){
        printf("Block ID %d is pendant\n", blockIdx.x);
    }
    // My child won't be set unless this block represents a valid pendant path
    // Could be a shared var
    int myChild = global_pendant_child_dev_ptr[blockIdx.x];
    if (threadIdx.x == 0){
        printf("Block ID %d's child is %d\n", blockIdx.x, myChild);
    }
    // Beginning of group of TPB pendant children
    int myBlockOffset = (blockIdx.x / blockDim.x) * blockDim.x;
    int myBlockIndex = blockIdx.x % blockDim.x;
    int leafIndex = (blockIdx.x / blockDim.x);
    int leafValue = global_active_leaf_indices[leafIndex];
    extern __shared__ int childrenAndDuplicateStatus[];
    // Write all 32 pendant children to shared memory
    // This offset works because it isnt treewise global, just levelwise
    childrenAndDuplicateStatus[threadIdx.x] = global_pendant_child_dev_ptr[myBlockOffset + threadIdx.x];
    __syncthreads();
    if (blockIdx.x == 0){
        printf("Block ID %d's pendant[%d] is %d\n", blockIdx.x, threadIdx.x, childrenAndDuplicateStatus[threadIdx.x]);
    }
    // See if myChild is duplicated, 1 vs all comparison written to shared memory
    // Also, if it is duplicated, only process the largest index duplicate
    // If it isn't duplicated, process the child.

    // By the cardinality of rational numbers, 
    childrenAndDuplicateStatus[blockDim.x + threadIdx.x] = ((childrenAndDuplicateStatus[threadIdx.x] == myChild) 
                                                            && myBlockIndex < threadIdx.x);
    __syncthreads();
    if (blockIdx.x == 0){
        printf("Block ID %d's childrenAndDuplicateStatus[%d] is %d\n", blockIdx.x, threadIdx.x, childrenAndDuplicateStatus[blockDim.x + blockIdx.x]);
    }
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
        printf("Block ID %d made it past the return\n", blockIdx.x);


    int pathsOffset = leafIndex * 4;
    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int valsAndColsOffset = leafIndex * numberOfEdgesPerGraph;
    int degreesOffset = leafIndex * numberOfRows;
    int LB, UB, v, vLB, vUB;
    // Set out-edges
    LB = global_row_offsets_dev_ptr[rowOffsOffset + myChild];
    UB = global_row_offsets_dev_ptr[rowOffsOffset + myChild + 1]; 
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("block ID %d Set offsets in PPP\n", blockIdx.x);
        printf("\n");
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
        global_values_dev_ptr[valsAndColsOffset + edge] = 0;
    }

    if (threadIdx.x == 0){
            global_degrees_dev_ptr[degreesOffset + myChild] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0){
        printf("Block ID %d Finished out edges PPP\n", blockIdx.x);
        printf("\n");
    }  
    if (threadIdx.x == 0){
        printf("Block %d, leafValue %d, myChild removed %d\n", blockIdx.x, leafValue, myChild);
        printf("\n");
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
    if (threadIdx.x == 0){
        printf("Block ID %d Started ParallelIdentifyVertexDisjointNonPendantPaths\n", blockIdx.x);
        printf("\n");
    }

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

    // Write all 32 nonpendant paths to shared memory
    for (int start = threadIdx.x; start < blockDim.x*4; start += blockDim.x){
        pathsAndIndependentStatus[start] = global_paths_ptr[globalPathOffset + start];
    }
    if (threadIdx.x == 0){
        printf("Block ID %d threadIdx.x copied path into sm\n", blockIdx.x, threadIdx.x);
    }
    printf("Block ID %d path %d %s pendant\n", blockIdx.x, threadIdx.x, 
        global_pendant_path_bool_dev_ptr[globalPendantPathBoolOffset + threadIdx.x] ? "is" : "isn't");

    // Automatically include pendant  paths to set
    pathsAndIndependentStatus[pendPathBoolOffset + threadIdx.x] = 
        global_pendant_path_bool_dev_ptr[globalPendantPathBoolOffset + threadIdx.x];

    __syncthreads();
    if (threadIdx.x == 0){
        printf("Block ID %d threadIdx.x copied into sm\n", blockIdx.x, threadIdx.x);
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
        printf("Block ID %d threadIdx.x created adj matrix\n", blockIdx.x, threadIdx.x);
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
            printf("Row %d thread %d included %d\n", row, threadIdx.x, pathsAndIndependentStatus[setReductionOffset + threadIdx.x]);
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

__global__ void ParallelAssignMISToNodes(int * global_active_leaf_indices,
                                        int * global_set_inclusion_bool_ptr,
                                        int * global_reduced_set_inclusion_count_ptr){
    
    int leafIndex = global_active_leaf_indices[blockIdx.x];


    // |I| - The cardinality of the set.  If |I| = 0; we don't induce children
    // Else we will induce (3*|I| children), Each path induces 3 leaves.
    int leavesThatICanProcess = global_reduced_set_inclusion_count_ptr[leafIndex];
    printf("Block ID %d thread  %d %s can process %d leaves\n", blockIdx.x, threadIdx.x, leavesThatICanProcess);

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
    */
}

__global__ void ParallelProcessDegreeZeroVertices(
                            int numberOfRows,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_degrees_dev_ptr){

    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Entered ProcessDeg0\n");
        printf("\n");
    }
    int leafIndex = blockIdx.x;

    extern __shared__ int degreeZeroVertex[];

    int degreesOffset = leafIndex * numberOfRows;
    int numVertices = global_remaining_vertices_size_dev_ptr[leafIndex];
    int numVerticesRemoved = 0;
    //for (int iter = 0; iter < blockDim.x; iter += blockDim.x){
    //    degreeZeroVertex[threadIdx.x] = 0;
    //}
    //_syncthreads();
    for (int vertex = threadIdx.x; vertex < numVertices; vertex += blockDim.x){
        numVerticesRemoved = 0;
        printf("threadIdx.x %d, blockIdx.x %d, Vertex %d loop\n", threadIdx.x, blockIdx.x, vertex);
        if (threadIdx.x == 0 && blockIdx.x == 0){
            printf("degreesOffset %d \n", degreesOffset);
            printf("vertex %d \n", vertex);
            printf("global_remaining_vertices_dev_ptr[degreesOffset + vertex] %d \n", global_remaining_vertices_dev_ptr[degreesOffset + vertex]);
            printf("full %d \n", global_degrees_dev_ptr[degreesOffset + global_remaining_vertices_dev_ptr[degreesOffset + vertex]]);
        }
        degreeZeroVertex[threadIdx.x] = (0 == global_degrees_dev_ptr[degreesOffset + global_remaining_vertices_dev_ptr[degreesOffset + vertex]]);
        if (blockIdx.x == 0){
            printf("Vertex %d set degreeZeroVertex %d since degree is %d\n", vertex, degreeZeroVertex[threadIdx.x], global_degrees_dev_ptr[degreesOffset + global_remaining_vertices_dev_ptr[degreesOffset + vertex]]);
        }
        // Makes this entry INT_MAX if degree 0
        // Leaves unaltered if not degree 0
        global_remaining_vertices_dev_ptr[degreesOffset + vertex] += (INT_MAX - global_remaining_vertices_dev_ptr[degreesOffset + vertex])*degreeZeroVertex[threadIdx.x];
        if (blockIdx.x == 0){
            printf("Vertex %d set global_remaining_vertices_dev_ptr %d\n", vertex, global_remaining_vertices_dev_ptr[degreesOffset + vertex]);
        }
        
        int i = blockDim.x/2;
        __syncthreads();
        while (i != 0) {
            if (threadIdx.x < i){
                printf("degreeZeroVertex[%d] = %d + %d\n", threadIdx.x, degreeZeroVertex[threadIdx.x], degreeZeroVertex[threadIdx.x + i]);
                degreeZeroVertex[threadIdx.x] += degreeZeroVertex[threadIdx.x + i];
                degreeZeroVertex[threadIdx.x + i] = 0;
            }
            __syncthreads();
            i /= 2;
        }
        if (threadIdx.x == 0){
            numVerticesRemoved += degreeZeroVertex[threadIdx.x];
            printf("numVerticesRemoved %d\n", numVerticesRemoved);

        }
    }
    // Update remaining vert size
    // Now just need to sort those INT_MAX entries to the end of the array
    if (threadIdx.x == 0){
        printf("numVerticesRemoved %d\n", numVerticesRemoved);
        global_remaining_vertices_size_dev_ptr[leafIndex] -= numVerticesRemoved;
    }
}
/*
__global__ void ParallelActiveVertexPathOffsets(int * global_active_leaf_indices,
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
        global_offsets_buffer[bufferRowOffsOffset + iter] = (blockIdx.x * numberOfEdgesPerGraph) + global_row_offsets_dev_ptr[rowOffsOffset + iter];
        printf("global_offsets_buffer[bufferRowOffsOffset + %d] = %d + %d\n", iter, (blockIdx.x * numberOfEdgesPerGraph), global_row_offsets_dev_ptr[rowOffsOffset + iter]);

    }

    if(threadIdx.x == 0){
        printf("LevelAware RowOffs \n");
        for (int i = 0; i < numberOfRows+1; ++i){
            printf("global_offsets_buffer[%d] = %d  \n",i, global_offsets_buffer[bufferRowOffsOffset+i]);
        }
        printf("\n");
    }
}
*/

__global__ void ParallelCreateLevelAwareRowOffsets(
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_offsets_buffer,
                            int * global_set_inclusion_bool_ptr){

    int leafIndex = blockIdx.x;

    printf("LevelAware RowOffs blockIdx %d is running\n", blockIdx.x);
    printf("LevelAware RowOffs leaf index %d is running\n", leafIndex);

    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int bufferRowOffsOffset = blockIdx.x * (numberOfRows + 1);

    for (int iter = threadIdx.x; iter < numberOfRows+1; iter += blockDim.x){
        global_offsets_buffer[bufferRowOffsOffset + iter] = (blockIdx.x * numberOfEdgesPerGraph) + global_row_offsets_dev_ptr[rowOffsOffset + iter];
        printf("global_offsets_buffer[bufferRowOffsOffset + %d] = %d + %d\n", iter, (blockIdx.x * numberOfEdgesPerGraph), global_row_offsets_dev_ptr[rowOffsOffset + iter]);

    }

    if(threadIdx.x == 0){
        printf("LevelAware RowOffs \n");
        for (int i = 0; i < numberOfRows+1; ++i){
            printf("global_offsets_buffer[%d] = %d  \n",i, global_offsets_buffer[bufferRowOffsOffset+i]);
        }
        printf("\n");
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
__global__ void ParallelQuicksortWithDNF(
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_degrees_dev_ptr){

    int row = threadIdx.x;



    for (int iter = row; iter < numberOfRows; iter += blockDim.x){

    }

}

/*
__global__ void SerialProcessPendantEdge(
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_paths_ptr,
                            int * global_pendant_path_bool_dev_ptr){
    // Set out-edges
    for (int i = 0; i < 2; ++i){
        LB = global_row_offsets_dev_ptr[rowOffsOffset + children[i]];
        UB = global_row_offsets_dev_ptr[rowOffsOffset + children[i] + 1];    
        for (int edge = LB + threadIndex; edge < UB; edge += blockDim.x){
            global_values_dev_ptr[valsAndColsOffset + edge] = 0;
        }
    }
    __syncthreads();
    if (threadIndex == 0 && blockIdx.x == 0){
        printf("Block %d, levelOffset %d, leafIndex %d, children removed %d %d\n", blockIdx.x, levelOffset, leafIndex, children[0], children[1]);
        for (int i = 0; i < global_edges_left_to_cover_count[(leafIndex-1)/3]; ++i){
            printf("(%d, %d) ",global_columns_dev_ptr[valsAndColsOffset + i], global_values_dev_ptr[valsAndColsOffset + i]);
        }
        printf("\n");
    }
    // (u,v) is the form of edge pairs.  We are traversing over v's outgoing edges, 
    // looking for u as the destination and turning off that edge.
    // this may be more elegantly handled by 
    // (1) an associative data structure
    // (2) an undirected graph 
    // Parallel implementations of both of these need to be investigated.
    for (int i = 0; i < 2; ++i){
        LB = global_row_offsets_dev_ptr[rowOffsOffset + children[i]];
        UB = global_row_offsets_dev_ptr[rowOffsOffset + children[i] + 1];    // Set out-edges
        for (int edge = LB + threadIndex; edge < UB; edge += blockDim.x){
            v = global_columns_dev_ptr[valsAndColsOffset + edge];
            // guarunteed to only have one incoming and one outgoing edge connecting (x,y)
            vLB = global_row_offsets_dev_ptr[rowOffsOffset + v];
            vUB = global_row_offsets_dev_ptr[rowOffsOffset + v + 1];
            for (int outgoingEdgeOfV = vLB + threadIndex; 
                    outgoingEdgeOfV < vUB; 
                        outgoingEdgeOfV += blockDim.x){
                if (children[i] == global_columns_dev_ptr[valsAndColsOffset + outgoingEdgeOfV]){
                    // Set in-edge
                    global_values_dev_ptr[valsAndColsOffset + outgoingEdgeOfV] = 0;
                }
            }
        }
    }
    __syncthreads();
}
*/
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

// Single threaded version
// DFS is implicitly single threaded
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
                                int * global_outgoing_edge_vertices,
                                int * global_outgoing_edge_vertices_count){
                             
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    if(threadID > 0) return;
    printf("Thread %d starting", threadID);

    int pathsOffset = leafIndex * 4;
    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int valsAndColsOffset = leafIndex * numberOfEdgesPerGraph;
    int degreesOffset = leafIndex * numberOfRows;
    int outgoingEdgeOffset = leafIndex * maxDegree;

    printf("Thread %d, pathsOffset : %d", threadID, pathsOffset);
    printf("Thread %d, rowOffsOffset : %d", threadID, rowOffsOffset);
    printf("Thread %d, valsAndColsOffset %d: ", threadID, valsAndColsOffset);
    printf("Thread %d, degreesOffset : %d", threadID, degreesOffset);

// Get random vertex

    unsigned int counter = 0;
    ulong seed = 0;
    int randomNumber = randomGPU(counter, leafIndex, seed);
    int randomIndex = randomNumber % global_vertices_remaining_count[leafIndex];
    int randomVertex = global_vertices_remaining[degreesOffset+randomIndex];
    printf("Thread %d, randomVertex : %d", threadID, randomVertex);
// dfs 
    for (int i = 0; i < 4; ++i){
        global_paths_ptr[pathsOffset + i] = randomVertex;
        printf("Thread %d, global_paths_ptr[pathsOffset + %d] : %d", threadID, i, global_paths_ptr[pathsOffset + i]);

        if (randomVertex == -1)
            break;
        global_outgoing_edge_vertices_count[leafIndex] = 0;
        for (int j = global_row_offsets_dev_ptr[rowOffsOffset + randomVertex]; 
                j < global_row_offsets_dev_ptr[rowOffsOffset + randomVertex + 1]; ++j){
            printf("Thread %d, global_values_dev_ptr[valsAndColsOffset + %d] : %d\n", threadID, j, global_values_dev_ptr[valsAndColsOffset + j]);
            if (global_values_dev_ptr[valsAndColsOffset + j] == 0)
                continue;
            else {
                global_outgoing_edge_vertices[outgoingEdgeOffset+global_outgoing_edge_vertices_count[leafIndex]] = j;
                printf("Thread %d, global_outgoing_edge_vertices[outgoingEdgeOffset+%d] : %d\n", threadID, global_outgoing_edge_vertices_count[leafIndex], global_outgoing_edge_vertices[outgoingEdgeOffset+global_outgoing_edge_vertices_count[leafIndex]]);
                ++global_outgoing_edge_vertices_count[leafIndex];
            }
        }
        ++counter;
        randomNumber = randomGPU(counter, leafIndex, seed);
        randomIndex = randomNumber % global_outgoing_edge_vertices_count[leafIndex];
        randomVertex = global_outgoing_edge_vertices[outgoingEdgeOffset+randomIndex];
        
        if (i > 0 && randomVertex == global_paths_ptr[pathsOffset + i - 1]){
            if (global_degrees_dev_ptr[degreesOffset+randomVertex] > 1){
                while(randomVertex == global_paths_ptr[pathsOffset + i - 1]){
                    ++counter;
                    randomNumber = randomGPU(counter, leafIndex, seed);
                    randomIndex = randomNumber % global_outgoing_edge_vertices_count[leafIndex];
                    randomVertex = global_outgoing_edge_vertices[outgoingEdgeOffset+randomIndex];
                }
            } else {
                randomVertex = -1;
            }
        }
    }
}

// Fill a perfect 3-ary tree to a given depth
__global__ void PopulateTreeParallelLevelWise_GPU(int numberOfLevels, 
                                                long long edgesPerNode,
                                                long long numberOfVertices,
                                                int * new_row_offsets_dev,
                                                int * new_columns_dev,
                                                int * values_dev,
                                                int * new_degrees_dev){

    long long myLevel = blockIdx.x;

    if (myLevel >= numberOfLevels)
        return;

    long long myLevelSize;
    long long levelOffset;
    if (myLevel != 0){
        myLevelSize = pow(3.0, myLevel-1);
        levelOffset = CalculateLevelOffset(myLevel);
    } else {
        myLevelSize = 1;
        levelOffset = 0;
    }

    long long leafIndex = threadIdx.x;

    for (int node = leafIndex; node < myLevelSize; node += blockDim.x){
        //graphs[levelOffset + node] = new Graph_GPU(g);
        printf("Thread %lu, block %lu", leafIndex, myLevel);

    }
}



void CallPopulateTree(int numberOfLevels, 
                    Graph & g){


    int maxDegree = g.GetLargestDegree();

    //int treeSize = 200000;
    int counters = 2;
    numberOfLevels = 2;
    int secondDeepestLevelSize = CalculateDeepestLevelWidth(numberOfLevels-2);;
    int deepestLevelSize = CalculateDeepestLevelWidth(numberOfLevels-1);;
    long long treeSize = CalculateSpaceForDesiredNumberOfLevels(numberOfLevels);
    long long bufferSize = deepestLevelSize;
    int expandedData = g.GetEdgesLeftToCover();
    int condensedData = g.GetVertexCount();
    int condensedData_plus1 = condensedData + 1;
    long long sizeOfSingleGraph = expandedData*2 + 2*condensedData + condensedData_plus1 + maxDegree + counters;
    long long totalMem = sizeOfSingleGraph * treeSize * sizeof(int) + 
        condensedData * bufferSize * sizeof(int) +
            2 * expandedData * bufferSize * sizeof(int);

    std::vector<std::set<int>> pendantChildren(treeSize);
    int pendantChild;

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
    do 
    {
        std::cout << '\n' << "Press enter to continue...; ctrl-c to terminate";
    } while (std::cin.get() != '\n');

// Each of these will be wrapped in a cub double buffer
// which will switch the active set passed to the methods which
// rely on these pointers.  This way I only keep the relevant 
// data instead of the entire tree.
    int * global_row_offsets_dev_ptr;
    int * global_columns_dev_ptr;
    int * global_values_dev_ptr;
    int * global_degrees_dev_ptr; 
    int * global_remaining_vertices_dev_ptr;

    int * global_row_offsets_dev_ptr_buffer;
    int * global_columns_dev_ptr_buffer;
    int * global_values_dev_ptr_buffer;
    int * global_degrees_dev_ptr_buffer; 
    int * global_remaining_vertices_dev_ptr_buffer;



    int * global_paths_ptr;
    int * global_remaining_vertices_size_dev_ptr;
    int * global_pendant_path_bool_dev_ptr;
    int * global_pendant_path_reduced_bool_dev_ptr;
    int * global_pendant_child_dev_ptr;
    //
    int * global_paths_indices_ptr, * global_paths_indices_ptr_buffer; 
    int * global_set_inclusion_bool_ptr, * global_set_inclusion_bool_ptr_buffer;
    int * global_set_path_offsets;
    //
    int * global_reduced_set_inclusion_count_ptr;
    int * global_paths_length;
    int * global_edges_left_to_cover_count;
    int * global_active_leaf_indices;
    int * global_memcpy_boolean;
    int * global_last_full_parent_vertex;

    int global_active_leaf_indices_count;

    int * global_column_buffer;
    int * global_vertex_buffer;
    int * global_value_buffer;
    int * global_offsets_buffer;
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
    cudaMalloc( (void**)&global_remaining_vertices_dev_ptr, (numberOfRows*deepestLevelSize) * sizeof(int) );

    cudaMalloc( (void**)&global_row_offsets_dev_ptr_buffer, ((numberOfRows+1)*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_columns_dev_ptr_buffer, (numberOfEdgesPerGraph*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_values_dev_ptr_buffer, (numberOfEdgesPerGraph*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_degrees_dev_ptr_buffer, (numberOfRows*deepestLevelSize) * sizeof(int) );
    cudaMalloc( (void**)&global_remaining_vertices_dev_ptr_buffer, (numberOfRows*deepestLevelSize) * sizeof(int) );

    cub::DoubleBuffer<int> row_offsets(global_row_offsets_dev_ptr, global_row_offsets_dev_ptr_buffer);
    cub::DoubleBuffer<int> columns(global_columns_dev_ptr, global_columns_dev_ptr_buffer);
    cub::DoubleBuffer<int> values(global_values_dev_ptr, global_values_dev_ptr_buffer);
    cub::DoubleBuffer<int> degrees(global_degrees_dev_ptr, global_degrees_dev_ptr_buffer);
    cub::DoubleBuffer<int> remaining_vertices(global_remaining_vertices_dev_ptr, global_remaining_vertices_dev_ptr_buffer);

    cudaMalloc( (void**)&global_paths_ptr, (max_dfs_depth*secondDeepestLevelSize*threadsPerBlock) * sizeof(int) );
    
    // Not global to the entire tree, overwritten every level
    // For sorting the threadsPerBlock paths by set inclusion in the MIS
    cudaMalloc( (void**)&global_paths_indices_ptr, secondDeepestLevelSize*threadsPerBlock*sizeof(int));
    cudaMalloc( (void**)&global_paths_indices_ptr_buffer, secondDeepestLevelSize*threadsPerBlock*sizeof(int));
    cudaMalloc( (void**)&global_set_inclusion_bool_ptr, threadsPerBlock * secondDeepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_set_inclusion_bool_ptr_buffer, threadsPerBlock * secondDeepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_set_path_offsets, (secondDeepestLevelSize+1) * sizeof(int) );

    
    // Not global to the entire tree, overwritten every level

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int> paths_indices(global_paths_indices_ptr, global_paths_indices_ptr_buffer);
    cub::DoubleBuffer<int> set_inclusion(global_set_inclusion_bool_ptr, global_set_inclusion_bool_ptr_buffer);

    cudaMemset(global_remaining_vertices_dev_ptr, INT_MAX, (numberOfRows*deepestLevelSize) * sizeof(int));

    // Since we statically allocate vertices remaining
    cudaMalloc( (void**)&global_vertex_segments, (deepestLevelSize+1) * sizeof(int) );

/*
    cudaMalloc( (void**)&global_column_buffer, numberOfEdgesPerGraph * deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_value_buffer, numberOfEdgesPerGraph * deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_vertex_buffer, numberOfRows * deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_offsets_buffer, (numberOfRows+1) * deepestLevelSize * sizeof(int) );

    cudaMalloc( (void**)&global_paths_length, treeSize * sizeof(int) );
*/

    cudaMalloc( (void**)&global_remaining_vertices_size_dev_ptr, treeSize * sizeof(int) );
    // Each node can process tPB pendant edges per call
    cudaMalloc( (void**)&global_pendant_path_bool_dev_ptr, threadsPerBlock * deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_pendant_path_reduced_bool_dev_ptr, deepestLevelSize * sizeof(int) );
    // Not global to the entire tree, overwritten every level
    cudaMalloc( (void**)&global_pendant_child_dev_ptr, threadsPerBlock * deepestLevelSize * sizeof(int) );


    cudaMalloc( (void**)&global_edges_left_to_cover_count, treeSize * sizeof(int) );

    // These two arrays direct the flow of the graph building process
    // If a vertex is not skipped and it has 0 paths produced, it needs to enter the DFS method
    // If a vertex is skipped, it doesn't do anything
    // If a vertex is not skipped and it has n > 0 paths produced, the largest number of
    // levels are induced, at least 1.
    cudaMalloc( (void**)&global_reduced_set_inclusion_count_ptr, treeSize * sizeof(int) );
    // For now I won't anticipate the decisions of deeper levels
    // Therefore, I can only work on a range
    // 0 < x < deepest level size vertices
    // x <= second deepest level size vertices
    // The minimum number of active leaves which produce a full tree is floor(log(threadsPerBlock)/log(3.0))
    // but we have to assume the worst, that is the entire second deepest level is full
    // and the last level is filled by inducing 3 children per leaf node

    cudaMalloc( (void**)&global_active_leaf_indices, activeLeavesPerNode*secondDeepestLevelSize*sizeof(int) );
    // If we want to use a serial list creation of active vertices
    //cudaMalloc( (void**)&global_active_leaf_indices, deepestLevelSize*sizeof(int) );

    // If the cols, vals, remaining vertices, need to be memcpied
    // When inducing a child, is true
    // If a DFS only produced pendants and another round of DFS take place
    // it is false 
    cudaMalloc( (void**)&global_memcpy_boolean, secondDeepestLevelSize * sizeof(int) );
    // Since we skip internal nodes, each active leaf needs to know the parent
    // from which cols, vals, remaining vertices are to be copied
    cudaMalloc( (void**)&global_last_full_parent_vertex, secondDeepestLevelSize * sizeof(int) );


    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    CopyGraphToDevice(g,
                    row_offsets.Current(),
                    columns.Current(),
                    values.Current(),
                    degrees.Current(),
                    numberOfEdgesPerGraph,
                    global_edges_left_to_cover_count,
                    remaining_vertices.Current(),
                    global_remaining_vertices_size_dev_ptr,
                    verticesRemainingInGraph,
                    global_active_leaf_indices);

    long long levelOffset = 0;
    long long levelUpperBound;
    int numberOfBlocksForOneThreadPerLeaf;
    numberOfLevels = 2;
    bool pendantNodeExists = true;

    int * pendantBools = new int[deepestLevelSize*threadsPerBlock];
    int * pendantReducedBools = new int[deepestLevelSize];
    int * pendantChildrenOfLevel = new int[deepestLevelSize*threadsPerBlock];

    int * nonpendantReducedCount = new int[deepestLevelSize];

    int * activeFlags = new int[treeSize];

    // One greater than secondDeepestLevelSize
    int dLSPlus1 = (deepestLevelSize + 1);
    int ceilOfDLSPlus1 = (dLSPlus1 + threadsPerBlock - 1) / threadsPerBlock;

    // Create Segment Offsets for RemainingVertices
    SetVerticesRemaingSegements<<<ceilOfDLSPlus1,threadsPerBlock>>>(deepestLevelSize,
                                                                    numberOfRows,
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
    while(activeVerticesCount){
    //for (int level = 0; level < numberOfLevels; ++level){
        //levelUpperBound = CalculateLevelUpperBound(level);
        //numberOfBlocksForOneThreadPerLeaf = ((levelUpperBound - levelOffset) + threadsPerBlock - 1) / threadsPerBlock;
        
        // For all but the first instance I need to sort...
        /*
        if(notFirstCall){
            RestoreDataStructuresAfterRemovingChildrenVertices(levelUpperBound,
                                                                levelOffset,
                                                                threadsPerBlock,
                                                                numberOfRows,
                                                                numberOfEdgesPerGraph,
                                                                global_row_offsets_dev_ptr,
                                                                global_offsets_buffer,
                                                                global_column_buffer,
                                                                global_value_buffer,
                                                                global_vertex_buffer,
                                                                global_vertex_segments,
                                                                global_remaining_vertices_dev_ptr,
                                                                global_columns_dev_ptr,
                                                                global_values_dev_ptr);
        }
        */
    
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
                            global_active_leaf_indices,
                            row_offsets.Current(),
                            columns.Current(),
                            remaining_vertices.Current(),
                            global_remaining_vertices_size_dev_ptr,
                            degrees.Current(),
                            global_paths_ptr,
                            paths_indices.Current(),
                            global_pendant_path_bool_dev_ptr,
                            global_pendant_path_reduced_bool_dev_ptr,
                            global_pendant_child_dev_ptr);
        
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        pendantNodeExists = false;
        cudaMemcpy(pendantBools, global_pendant_path_bool_dev_ptr, threadsPerBlock*(activeVerticesCount)*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pendantReducedBools, global_pendant_path_reduced_bool_dev_ptr, (activeVerticesCount)*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pendantChildrenOfLevel, global_pendant_child_dev_ptr, threadsPerBlock*(activeVerticesCount)*sizeof(int), cudaMemcpyDeviceToHost);

        for (int node = levelOffset; node < levelUpperBound; ++node){
            // global_pendant_path_bool_dev_ptr was defined as an OR of 
            // 0) path[0] == path[2]
            // 1) path[1] == path[3]
            std::cout << "node " << node << std::endl;
            std::cout << "global_pendant_path_bool_dev_ptr[node] " << pendantBools[node] << std::endl;

            std::cout << "!global_pendant_path_bool_dev_ptr[node] " << !pendantBools[node] << std::endl;

            if (pendantReducedBools[node]){
                std::cout << "node " << node << " contains a pendant edge" << std::endl;
                for (int pendantPathIndex = 0; pendantPathIndex < threadsPerBlock; ++pendantPathIndex){
                    if(pendantBools[pendantPathIndex]){
                        pendantChild = pendantChildrenOfLevel[(node - levelOffset)*threadsPerBlock + pendantPathIndex];
                        pendantChildren[node].insert(pendantChild);
                        std::cout << "node " << node << "'s pendantChild " << pendantChild << " was pushed" << std::endl;
                    }
                }
            }
        }
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
        // Each node assigned threadsPerBlock blocks,  
        // Up to threadsPerBlock pendant children are processed
        // 1 pendant child per block
        // outgoing and incoming edges of the pendant child 
        // are processed at thread level
        // Block immediately returns if nonpendant child 
        // or duplicate pendant child and not the largest
        // indexed instance of that child
        ParallelProcessPendantEdges<<<(activeVerticesCount)*threadsPerBlock,
                                    threadsPerBlock,
                                    2*threadsPerBlock*sizeof(int)>>>
                        (numberOfRows,
                        numberOfEdgesPerGraph,
                        global_active_leaf_indices,
                        row_offsets.Current(),
                        columns.Current(),
                        values.Current(),
                        degrees.Current(),
                        global_pendant_path_bool_dev_ptr,
                        global_pendant_child_dev_ptr);
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        ParallelProcessDegreeZeroVertices<<<activeVerticesCount,
                                            threadsPerBlock,
                                            threadsPerBlock*sizeof(int)>>>
                        (numberOfRows,
                        remaining_vertices.Current(),
                        global_remaining_vertices_size_dev_ptr,
                        degrees.Current());
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        ParallelIdentifyVertexDisjointNonPendantPaths<<<activeVerticesCount,
                                                        threadsPerBlock,
                                                        (threadsPerBlock*threadsPerBlock + 
                                                        10*threadsPerBlock)*sizeof(int)>>>
                                                        (numberOfRows,
                                                        numberOfEdgesPerGraph,
                                                        row_offsets.Current(),
                                                        columns.Current(),
                                                        values.Current(),
                                                        global_pendant_path_bool_dev_ptr,
                                                        global_paths_ptr,
                                                        set_inclusion.Current(),
                                                        global_reduced_set_inclusion_count_ptr);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        SortPathIndices(activeVerticesCount,
                        threadsPerBlock,
                        paths_indices,
                        set_inclusion,
                        global_set_path_offsets);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        PrintSets<<<1,1>>>(activeVerticesCount,
                paths_indices.Current(),
                paths_indices.Alternate(),
                set_inclusion.Current(),
                set_inclusion.Alternate(),
                global_set_path_offsets);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        // Just to test a single iteration
        activeVerticesCount = 0;

    }
/*
        cudaMemcpy(activeFlags, global_active_vertices, treeSize*sizeof(int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        std::cout << "TREE" << std::endl;
        for (int level = 0; level < numberOfLevels; ++level){
            int levelOffsetForPrinting = CalculateLevelOffset(level);
            int levelUpperBoundForPrinting = CalculateLevelUpperBound(level);
            for (int i = levelOffsetForPrinting; i < levelUpperBoundForPrinting; ++i)
                std::cout << activeFlags[i];
            std::cout << std::endl;
        }
*/

        /*
        // We need the number of children for each leaf to induce.
        cudaMemcpy(nonpendantReducedCount, global_reduced_set_inclusion_count_ptr, (levelUpperBound - levelOffset)*sizeof(int), cudaMemcpyDeviceToHost);

        // For now I just copy the entire graph to all children
        // My custom written induce subgraph method uses dynamically
        // allocated arrays which are unpredictable.  The memory is already 
        // allocated for the entire tree, and copying the entire graph likely
        // doesnt effect runtime since d2d bandwith is very high.
        for (int leafIndex = levelOffset; leafIndex <  levelUpperBound; ++leafIndex){
            cudaMemcpy(global_row_offsets_dev_ptr, global_row_offsets_dev_ptr, (numberOfRows+1)*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(global_columns_dev_ptr, global_columns_dev_ptr, numberOfEdgesPerGraph*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(global_values_dev_ptr, global_values_dev_ptr, numberOfEdgesPerGraph*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(global_degrees_dev_ptr, global_degrees_dev_ptr, numberOfRows*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(global_remaining_vertices_dev_ptr, global_remaining_vertices_dev_ptr, numberOfRows*sizeof(int), cudaMemcpyDeviceToDevice);
        }

        notFirstCall = true;
        */
    

    for (const auto& inner: pendantChildren) { // auto is std::vector<int>
        for (auto e: inner) { // auto is int
            std::cout << e << " ";
        }
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    cudaFree( global_row_offsets_dev_ptr );
    cudaFree( global_columns_dev_ptr );
    cudaFree( global_values_dev_ptr );
    cudaFree( global_degrees_dev_ptr );
    cudaFree( global_paths_ptr );
    cudaFree( global_paths_length );
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
                        int * global_active_leaf_indices){

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
    cudaMemset(global_active_leaf_indices, 1, 1*sizeof(int));
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


// Logic of the tree
    // Every level decreases the number of remaining vertices by at least 2
    // more sophisticate analysis could be performed by analyzing the graph
    // i.e. number of degree 1 vertices, (case 3) - a level decreases by > 2
    // number of degree 2 vertices with a pendant edge (case 2) - a level decreases by > 2
    // number of triangles in a graph (case 1)
    // gPrime is at root of tree
    // This is a 3-ary tree, m = 3
    // if a node has an index i, its c-th child in range {1,…,m} 
    // is found at index m ⋅ i + c, while its parent (if any) is 
    // found at index floor((i-1)/m).

// This method benefits from more compact storage and 
// better locality of reference, particularly during a 
// preorder traversal. The space complexity of this 
// method is O(m^n).  Actually smaller - TODO
// calculate by recursion tree

    // We are setting parent pointers, in case we find space
    // to be a constraint, we are halfway to dynamic trees,
    // we just need to pop a free graph object off a queue 
    // and induce.  
    // We may have no use for iterating over a graph from the root.
/*
__host__ __device__ void PopulateTree(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer){
    // ceiling(vertexCount/2) loops
    int result, childVertex;
    for (int i = 0; i < treeSize; ++i){
        result = GenerateChildren(graphs[i]);
        while (graphs[i].GetChildrenVertices().size() == 1){
            graphs[i].ProcessImmediately(graphs[i].GetChildrenVertices().front());
            graphs[i].GetChildrenVertices().clear();
            result = GenerateChildren(graphs[i]);
        }       
        if (result == -1){
            TraverseUpTree(i, graphs, answer);
            return;
        } else {
            for (int c = 1; c <= 3; ++c){
                std::cout << "i : " << i << ", c : " << c << std::endl;
                graphs[3*i + c].InitGPrime(graphs[i], graphs[i].GetChildrenVertices()[c-1]);
            }
        }
    }
}

// Fill a perfect 3-ary tree to a given depth
__host__ __device__ int PopulateTreeParallelLevelWise(int numberOfLevels, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer){
    // ceiling(vertexCount/2) loops
    volatile bool flag=false;
    std::vector<long long> resultsFlags;
    long long maximumLevelSize = pow(3.0, numberOfLevels-1);
    resultsFlags.reserve(maximumLevelSize);
    long long leafIndex;
    long long levelOffset = 0;
    long long upperBound = 0;
    long long previousLevelSize = 0;
    long long thisLevelSize = 0;
    long long count = 0;
    for (int level = 0; level < numberOfLevels; ++level){
        // level 0 - [0,1); lvlOff = 0 + 0
        // level 1 - [1,4); lvlOff = 0 + 3^0 = 1
        // level 2 - [4,13);lvlOff = 1 + 3^1 = 4
        if (level != 0){
            previousLevelSize = thisLevelSize;
            levelOffset += previousLevelSize;
        }
        thisLevelSize = pow(3.0, level);
        upperBound = levelOffset + thisLevelSize;
        
        resultsFlags.clear();
        for (count = levelOffset; count < upperBound; ++count)
            resultsFlags.push_back(-1);

//        #pragma omp parallel for default(none) \
//            shared(graphs, levelOffset, level, numberOfLevels, upperBound, flag, resultsFlags) \
//            private (leafIndex)

        for (leafIndex = levelOffset; leafIndex < upperBound; ++leafIndex){
            
            // Allows for pseudo-early termination if an answer is found
            // All iterations which havent begun will terminate quickly
    //        if(flag) continue;

            if (flag) continue;
            int result;
            result = GenerateChildren(graphs[leafIndex]);
            if (result == -1)
            {
                flag = true;
                resultsFlags[leafIndex - levelOffset] = leafIndex;
            }
            // This is a strict 3-ary tree
            while (graphs[leafIndex].GetChildrenVertices().size() == 1){
                graphs[leafIndex].ProcessImmediately(graphs[leafIndex].GetChildrenVertices().front());
                graphs[leafIndex].GetChildrenVertices().clear();
                result = GenerateChildren(graphs[leafIndex]);
                if (result == -1)
                {
                    flag = true;
                    resultsFlags[leafIndex - levelOffset] = leafIndex;
                }  
            }
            // We dont initiate the last level and we stop if we cant make more children 
            if (level + 1 != numberOfLevels && result != -1)
                for (int c = 1; c <= 3; ++c){
                    printf("level : %d, level offset : %lld, leafIndex : %lld, c : %d\n", level, levelOffset, leafIndex, c);
                    graphs[3*leafIndex + c].InitGPrime(graphs[leafIndex], graphs[leafIndex].GetChildrenVertices()[c-1]);
                }
        }
        if (flag)
            for(auto & v : resultsFlags)
                if (v != -1)
                    return v;
    }
    return -1;
}
// This method can be rewritten to use fill all Graphs allocated
// Irrespective of whether the last level is full
__host__ __device__ void PopulateTreeParallelAsymmetric(int treeSize, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer){
    // ceiling(vertexCount/2) loops
    int numberOfLevels = int(ceil(log(treeSize) / log(3)));
    int leafIndex;
    int levelOffset = 0;
    int upperBound = 0;
    for (int level = 0; level < numberOfLevels; ++level){
        // level 0 - [0,1); lvlOff = 0 + 0
        // level 1 - [1,4); lvlOff = 0 + 3^0 = 1
        // level 2 - [4,13);lvlOff = 1 + 3^1 = 4
        if (level != 0)
            levelOffset += int(pow(3.0, level-1));
        if (level + 1 != numberOfLevels){
            upperBound = levelOffset + int(pow(3.0, level));
        } else {
            upperBound = treeSize;
        }
        #pragma omp parallel for default(none) \
                            shared(treeSize, graphs, levelOffset, level, upperBound) \
                            private (leafIndex)
        for (leafIndex = levelOffset; leafIndex < upperBound; ++leafIndex){
            int result;
            result = GenerateChildren(graphs[leafIndex]);
            // This is a strict 3-ary tree
            while (graphs[leafIndex].GetChildrenVertices().size() == 1){
                graphs[leafIndex].ProcessImmediately(graphs[leafIndex].GetChildrenVertices().front());
                graphs[leafIndex].GetChildrenVertices().clear();
                result = GenerateChildren(graphs[leafIndex]);
            }       
            for (int c = 1; c <= 3; ++c){
                if (3*leafIndex + c < treeSize){
                    printf("level : %d, level offset : %d, leafIndex : %d, c : %d\n", level, levelOffset, leafIndex, c);
                    graphs[3*leafIndex + c].InitGPrime(graphs[leafIndex], graphs[leafIndex].GetChildrenVertices()[c-1]);
                }
            }
        }
    }
}

__host__ __device__ int GenerateChildren(Graph & child_g){

    std::vector< std::vector<int> > & childrensVertices_ref = child_g.GetChildrenVertices();

    std::vector<int> path;
    int randomVertex = GetRandomVertex(child_g.GetRemainingVerticesRef());
    std::cout << "Grabbing a randomVertex: " <<  randomVertex<< std::endl;
    if(randomVertex == -1)
        return randomVertex;

    path.push_back(randomVertex);

    DFS(child_g.GetCSR().GetNewRowOffRef(), 
        child_g.GetCSR().GetNewColRef(), 
        child_g.GetCSR().GetNewValRef(),
        path, 
        randomVertex);

    for (auto & v : path){
        std::cout << v << " ";
        if (v < 0 || v > child_g.GetVertexCount())
            std::cout << "error" << std::endl;
    }
    std::cout << std::endl;
    int caseNumber = classifyPath(path);
    std::cout << "Case number: " << caseNumber << std::endl;
    createVertexSetsForEachChild(childrensVertices_ref, caseNumber, path);
    for (auto & vv : childrensVertices_ref)
        for (auto & v : vv)
            if (v < 0 || v > child_g.GetVertexCount())
                std::cout << "error" << std::endl;

    return 0;
}

__host__ __device__ int GetRandomVertex(std::vector<int> & verticesRemaining){
    if(verticesRemaining.size() == 0)
        return -1;
    int index = rand() % verticesRemaining.size();
    return verticesRemaining[index];
}


// DFS of maximum length 3. No simple cycles u -> v -> u 
__host__ __device__ void DFS(std::vector<int> & new_row_off,
                    std::vector<int> & new_col_ref, 
                    std::vector<int> & new_vals_ref,
                    std::vector<int> & path, 
                    int rootVertex){
    if (path.size() == 4)
        return;

    int randomOutgoingEdge = GetRandomOutgoingEdge(new_row_off, new_col_ref, new_vals_ref, rootVertex, path);
    if (randomOutgoingEdge < 0) {
        std::cout << "terminate DFS" << std::endl;
        return;
    } else {
        path.push_back(randomOutgoingEdge);
        return DFS(new_row_off, new_col_ref, new_vals_ref, path, randomOutgoingEdge);
    }
}

__host__ __device__ int GetRandomOutgoingEdge(  std::vector<int> & new_row_off,
                                        std::vector<int> & new_col_ref,
                                        std::vector<int> & new_values_ref,
                                        int v, 
                                        std::vector<int> & path){

    std::vector<int> outgoingEdges(&new_col_ref[new_row_off[v]],
                        &new_col_ref[new_row_off[v+1]]);

    std::vector<int> outgoingEdgeValues(&new_values_ref[new_row_off[v]],
                    &new_values_ref[new_row_off[v+1]]);

    std::vector<std::pair<int, int>> edgesAndValues;
    edgesAndValues.reserve(outgoingEdges.size());
    std::transform(outgoingEdges.begin(), outgoingEdges.end(), outgoingEdgeValues.begin(), std::back_inserter(edgesAndValues),
               [](int a, int b) { return std::make_pair(a, b); });

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(edgesAndValues.begin(), edgesAndValues.end(), g);
    std::vector< std::pair<int,int> >::iterator it = edgesAndValues.begin();

    while (it != edgesAndValues.end()){
        // To prevent simple paths, must at least have 2 entries, 
        //assuming there are no self edges, since the first entry, v,
        //is randomly chosen and the second entry is a random out edge 
        if (path.size() > 1 && it->first == path.rbegin()[1]  || it->second == 0) {
            //std::cout << "Wouldve been a simple path, skipping " << *it << std::endl;
            ++it;
        } else
            return it->first;
    }

    return -1;
}


__host__ __device__ int classifyPath(std::vector<int> & path){
    if (path.size()==2)
        return 3;
    else if (path.size()==3)
        return 2;
    else if (path.front() == path.back())
        return 1;
    else
        return 0;
}

__host__ __device__ void createVertexSetsForEachChild(std::vector< std::vector<int> > & childrensVertices,
                                            int caseNumber, 
                                            std::vector<int> & path){
    if (caseNumber == 0) {
        // 3 Children 
        childrensVertices.resize(3);
        // Each with 2 vertices 
        for (auto & cV : childrensVertices)
            cV.reserve(2);
        childrensVertices[0].push_back(path[0]);
        childrensVertices[0].push_back(path[2]);

        childrensVertices[1].push_back(path[1]);
        childrensVertices[1].push_back(path[2]);

        childrensVertices[2].push_back(path[1]);
        childrensVertices[2].push_back(path[3]);

    } else if (caseNumber == 1) {

        // 3 Children 
        childrensVertices.resize(3);
        // Each with 2 vertices 
        for (auto & cV : childrensVertices)
            cV.reserve(2);
        childrensVertices[0].push_back(path[0]);
        childrensVertices[0].push_back(path[1]);

        childrensVertices[1].push_back(path[1]);
        childrensVertices[1].push_back(path[2]);

        childrensVertices[2].push_back(path[0]);
        childrensVertices[2].push_back(path[2]);

    } else if (caseNumber == 2) {

        childrensVertices.resize(1);
        childrensVertices[0].reserve(1);
        childrensVertices[0].push_back(path[1]);

    } else {

        childrensVertices.resize(1);
        childrensVertices[0].reserve(1);
        childrensVertices[0].push_back(path[0]);

    }
}

__host__ __device__ void TraverseUpTree(int index, 
                                std::vector<Graph> & graphs,
                                std::vector<int> & answer){
    bool haventReachedRoot = true;
    while(haventReachedRoot) {
        if (index == 0)
            haventReachedRoot = false;
        for (auto & v : graphs[index].GetVerticesThisGraphIncludedInTheCover())
            answer.push_back(v);
        index = (index-1)/3;
    } 
}
*/

#endif
