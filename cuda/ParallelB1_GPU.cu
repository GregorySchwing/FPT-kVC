#ifdef FPT_CUDA

#include "ParallelB1_GPU.cuh"
#include <math.h>       /* pow */
#include "cub/cub.cuh"

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
                            int levelOffset,
                            int levelUpperBound,
                            int * global_row_offsets_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_degrees_dev_ptr,
                            int * global_edges_left_to_cover_count){

    int leafIndex = levelOffset + blockIdx.x;
    if (leafIndex >= levelUpperBound) return;

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
                            int * global_nonpendant_child_dev_ptr){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Entered DFS\n");
        printf("\n");
    }
    int leafIndex = levelOffset + blockIdx.x;
    if (leafIndex >= levelUpperBound)
        return;
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Set leafIndex\n");
        printf("\n");
    }
    // Initialized to 0, so will always perform DFS on first call
    // Subsequently, only perform DFS on pendant edges, so nonpendant false
    if (global_nonpendant_path_bool_dev_ptr[leafIndex])
        return;
    int globalPathOffset = leafIndex * 4;
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
        printf("remainingVerticesSize\n");
        printf("\n");
    }
    int outEdgesCount;
    r = randomGPU_four(counter, leafIndex, seed);
    // Random starting point
    pathsAndPendantStatus[sharedMemPathOffset + iteration] = r[iteration] % remainingVerticesSize;
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("pathsAndPendantStatus\n");
        printf("\n");
    }
    ++iteration;

    // Set random out at depth 1
    int randomVertRowOff = global_row_offsets_dev_ptr[rowOffsOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]];
    // Using degrees allow us to ignore the edges which have been turned off
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("randomVertRowOff %d\n", randomVertRowOff);
        printf("\n");
    }
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("degreesOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1] %d\n",degreesOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]);
    }
       if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("degreesOffset  %d\n",degreesOffset);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("pathsAndPendantStatus[sharedMemPathOffset + iteration - 1] %d\n",pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]);
    }
    outEdgesCount = global_degrees_dev_ptr[degreesOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1]];
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("outEdgesCount %d\n", outEdgesCount);
        printf("\n");
    }
    //outEdgesCount = global_row_offsets_dev_ptr[rowOffsOffset + pathsAndPendantStatus[sharedMemPathOffset + iteration - 1] + 1]
    //                - randomVertRowOff;
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("valsAndColsOffset + randomVertRowOff + (r[iteration] mod outEdgesCount) %d\n", valsAndColsOffset + randomVertRowOff + (r[iteration] % outEdgesCount));
    }
    pathsAndPendantStatus[sharedMemPathOffset + iteration] =  global_columns_dev_ptr[valsAndColsOffset + randomVertRowOff + (r[iteration] % outEdgesCount)];
    ++iteration;
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Block %d, levelOffset %d, leafIndex %d, got through first 2 iterations\n", blockIdx.x, levelOffset, leafIndex);
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
        // OutEdgesCount != 2 means there is another path that isn't a cycle
        if(pathsAndPendantStatus[sharedMemPathOffset + iteration] == 
            pathsAndPendantStatus[sharedMemPathOffset + iteration - 2]
                && outEdgesCount != 1){
            pathsAndPendantStatus[sharedMemPathOffset + iteration] =  global_columns_dev_ptr[valsAndColsOffset + randomVertRowOff + ((r[iteration] + 1) % outEdgesCount)];
        }
    }
    pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x] = (pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + 0] == pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + 2]);
    pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x] |= (pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + 1] == pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + 3]);
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Block %d, levelOffset %d, leafIndex %d, got through last 2 iterations\n", blockIdx.x, levelOffset, leafIndex);
        printf("\n");
    }
    printf("Thread %d is %s\n", threadIdx.x, pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x] ? "pendant" : "nonpendant");

    int i = blockDim.x/2;
    // Checks for any nonpendant edge path exists
    while (i != 0) {
        if (threadIdx.x < i){
            pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x] &= pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Write pendant status to global memory
    // We detected pendant, but we store the converse, nonpendantness
    if (threadIdx.x == 0){
        global_nonpendant_path_bool_dev_ptr[leafIndex] = !pathsAndPendantStatus[isInvalidPathBooleanArrayOffset];
        printf("leafIndex %d is %s\n", leafIndex, global_nonpendant_path_bool_dev_ptr[leafIndex] ? "nonpendant" : "pendant");
    }
    // A nonpendant exists
    if (!pathsAndPendantStatus[isInvalidPathBooleanArrayOffset]){
        // Regenerate pendant booleans
        pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x] = (pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + 0] == pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + 2]);
        pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + threadIdx.x] |= (pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + 1] == pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + 3]);
        // We give Case 3 priority over Case 2,
        // Since the serial algorithm short-circuits 
        // upon finding a pendant edge

        // We know either 
        // Case 3 - length 2
        // v, v1
        //path[0] == path[2], desired child is v
        // If path[0] == path[2] then path[0] != path[2]
        // Hence, cI == 0, since false casted to int is 0
        // Therefore, v == path[cI]
        int childIndex = global_paths_ptr[globalPathOffset + 0] != global_paths_ptr[globalPathOffset + 2];

        // or
        // Case 2 - length 3
        // v, v1, v2
        // if path[0] != path[2] was true, then path[1] == path[3]
        // cI == 1, since true casted to int is 1
        // Desired child is v1
        // Therefore, v1 == path[cI]
        global_nonpendant_child_dev_ptr[blockIdx.x] = global_paths_ptr[globalPathOffset + childIndex];
        if (threadIdx.x == 0){
            // One of these is guarunteed nonpendant
            for (int i = 0; i < blockDim.x; ++i){
                // Since these are random paths, using the first possible nonpendant isnt biased
                if (!pathsAndPendantStatus[isInvalidPathBooleanArrayOffset + i]){
                    for (int j = 0; j < 4; ++j)
                        global_paths_ptr[globalPathOffset + j] = pathsAndPendantStatus[i*4 + j];
                    return;
                }
            }
        }
    } else {
        if (threadIdx.x == 0){
            // Since these are random paths, using the first pendant isnt biased
            for (int j = 0; j < 4; ++j){
                global_paths_ptr[globalPathOffset + j] = pathsAndPendantStatus[j];
            }
            printf("leaf index %d, path (%d -> %d -> %d -> %d)\n", leafIndex, 
                global_paths_ptr[globalPathOffset + 0],
                global_paths_ptr[globalPathOffset + 1],
                global_paths_ptr[globalPathOffset + 2],
                global_paths_ptr[globalPathOffset + 3]);
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Block ID %d Finished DFS\n", blockIdx.x);
        printf("\n");
    }
}

__global__ void ParallelProcessPendantEdges(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_values_dev_ptr,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_degrees_dev_ptr,
                            int * global_paths_ptr,
                            int * global_nonpendant_path_bool_dev_ptr){

    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Block ID %d Started ParallelProcessPendantEdges\n", blockIdx.x);
        printf("\n");
    }
    int leafIndex = levelOffset + blockIdx.x;
    // Only process pendant edges
    if (global_nonpendant_path_bool_dev_ptr[leafIndex])
        return;
    int pathsOffset = leafIndex * 4;
    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int valsAndColsOffset = leafIndex * numberOfEdgesPerGraph;
    int degreesOffset = leafIndex * numberOfRows;
    int childIndex = global_paths_ptr[pathsOffset + 0] != global_paths_ptr[pathsOffset + 2];
    int child = global_paths_ptr[pathsOffset + childIndex];
    int LB, UB, v, vLB, vUB;
    // Set out-edges
    LB = global_row_offsets_dev_ptr[rowOffsOffset + child];
    UB = global_row_offsets_dev_ptr[rowOffsOffset + child + 1]; 
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("block ID %d Set offsets in PPP\n", blockIdx.x);
        printf("\n");
    }   
    for (int edge = LB + threadIdx.x; edge < UB; edge += blockDim.x){
        // Since there are only 2 edges b/w each node,
        // We can safely decrement the target node's degree
        global_degrees_dev_ptr[degreesOffset + 
            global_columns_dev_ptr[valsAndColsOffset + edge]] 
                -= global_values_dev_ptr[valsAndColsOffset + edge];
        global_values_dev_ptr[valsAndColsOffset + edge] = 0;
    }

    if (threadIdx.x == 0){
            global_degrees_dev_ptr[degreesOffset + child] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Block ID %d Finished out edges PPP\n", blockIdx.x);
        printf("\n");
    }  
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Block %d, levelOffset %d, leafIndex %d, child removed %d\n", blockIdx.x, levelOffset, leafIndex, child);
        printf("\n");
    }
    // (u,v) is the form of edge pairs.  We are traversing over v's outgoing edges, 
    // looking for u as the destination and turning off that edge.
    // this may be more elegantly handled by 
    // (1) an associative data structure
    // (2) an undirected graph 
    // Parallel implementations of both of these need to be investigated.
    bool foundChild, tmp;
    LB = global_row_offsets_dev_ptr[rowOffsOffset + child];
    UB = global_row_offsets_dev_ptr[rowOffsOffset + child + 1];    // Set out-edges
    for (int edge = LB + threadIdx.x; edge < UB; edge += blockDim.x){
        v = global_columns_dev_ptr[valsAndColsOffset + edge];
        // guarunteed to only have one incoming and one outgoing edge connecting (x,y)
        // All outgoing edges were set and are separated from this method by a __syncthreads
        // Thus there is no chance of decrementing the degree of the same node simulataneously
        vLB = global_row_offsets_dev_ptr[rowOffsOffset + v];
        vUB = global_row_offsets_dev_ptr[rowOffsOffset + v + 1];
        for (int outgoingEdgeOfV = vLB + threadIdx.x; 
                outgoingEdgeOfV < vUB; 
                    outgoingEdgeOfV += blockDim.x){

                foundChild = child == global_columns_dev_ptr[valsAndColsOffset + outgoingEdgeOfV];
                // Set in-edge
                // store edge status
                tmp = global_values_dev_ptr[valsAndColsOffset + outgoingEdgeOfV];
                //   foundChild     tmp   (foundChild & tmp)  (foundChild & tmp)^tmp
                //1)      0          0            0                       0
                //2)      1          0            0                       0
                //3)      0          1            0                       1
                //4)      1          1            1                       0
                //
                // Case 1: isnt child and edge is off, stay off
                // Case 2: is child and edge is off, stay off
                // Case 3: isn't child and edge is on, stay on
                // Case 4: is child and edge is on, turn off
                // All this logic is necessary because we aren't using degree to set upperbound
                // we are using row offsets, which may include some edges turned off on a previous
                // pendant edge processing step.
                global_values_dev_ptr[valsAndColsOffset + outgoingEdgeOfV] ^= (foundChild & tmp);
        
        }
    }
    __syncthreads();
}

__global__ void ParallelProcessDegreeZeroVertices(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_degrees_dev_ptr){

    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Entered ProcessDeg0\n");
        printf("\n");
    }
    int leafIndex = levelOffset + blockIdx.x;
    if (leafIndex >= levelUpperBound)
        return;    

    extern __shared__ int degreeZeroVertex[];

    int degreesOffset = leafIndex * numberOfRows;
    int vertexOffset = 0;
    int numVertices = global_remaining_vertices_size_dev_ptr[leafIndex];
    int numVerticesRemoved = 0;
    for (int vertex = vertexOffset + threadIdx.x; vertex < numVertices; vertexOffset += blockDim.x){
        // Reinitialize
        degreeZeroVertex[threadIdx.x] = 0;
        degreeZeroVertex[threadIdx.x] = (0 == global_degrees_dev_ptr[degreesOffset + global_remaining_vertices_dev_ptr[degreesOffset + vertex]]);
        // Makes this entry INT_MAX if degree 0
        // Leaves unaltered if not degree 0
        global_remaining_vertices_dev_ptr[degreesOffset + vertex] += (INT_MAX - global_remaining_vertices_dev_ptr[degreesOffset + vertex])*degreeZeroVertex[threadIdx.x];
        int i = blockDim.x/2;
        __syncthreads();
        // Checks for any nonpendant edge path exists
        while (i != 0) {
            if (threadIdx.x < i){
                degreeZeroVertex[threadIdx.x] += degreeZeroVertex[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }
        if (threadIdx.x == 0)
            numVerticesRemoved += degreeZeroVertex[threadIdx.x];
    }
    // Update remaining vert size
    // Now just need to sort those INT_MAX entries to the end of the array
    if (threadIdx.x == 0)
        global_remaining_vertices_size_dev_ptr[leafIndex] -= numVerticesRemoved;
}


__global__ void ParallelCreateLevelAwareRowOffsets(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_offsets_buffer){

    int leafIndex = levelOffset + blockIdx.x;
    if (leafIndex >= levelUpperBound)
        return;    

    int rowOffsOffset = leafIndex * (numberOfRows + 1);
    int bufferRowOffsOffset = blockIdx.x * (numberOfRows + 1);

    for (int iter = threadIdx.x; iter < numberOfRows+1; iter += blockDim.x){
        global_offsets_buffer[bufferRowOffsOffset + iter] = (blockIdx.x * numberOfEdgesPerGraph) + global_row_offsets_dev_ptr[rowOffsOffset + iter];
    }
}

__global__ void ParallelQuicksortWithDNF(int levelOffset,
                            int levelUpperBound,
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
__global__ void SerialProcessPendantEdge(int levelOffset,
                            int levelUpperBound,
                            int numberOfRows,
                            int numberOfEdgesPerGraph,
                            int * global_row_offsets_dev_ptr,
                            int * global_columns_dev_ptr,
                            int * global_remaining_vertices_dev_ptr,
                            int * global_remaining_vertices_size_dev_ptr,
                            int * global_paths_ptr,
                            int * global_nonpendant_path_bool_dev_ptr){
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

    std::vector<std::vector<int>> pendantChildren(treeSize);
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

    int * global_row_offsets_dev_ptr;
    int * global_columns_dev_ptr;
    int * global_values_dev_ptr;
    int * global_degrees_dev_ptr; 
    int * global_paths_ptr; 
    int * global_remaining_vertices_ptr;
    int * global_remaining_vertices_size_dev_ptr;
    int * global_nonpendant_path_bool_dev_ptr;
    int * global_nonpendant_child_dev_ptr;
    int * global_paths_length;
    int * global_edges_left_to_cover_count;

    int * global_column_buffer;
    int * global_vertex_buffer;
    int * global_value_buffer;
    int * global_offsets_buffer;

    int max_dfs_depth = 4;
    int numberOfRows = g.GetNumberOfRows();
    int numberOfEdgesPerGraph = g.GetEdgesLeftToCover(); 
    int verticesRemainingInGraph = g.GetRemainingVertices().size(); 

    cudaMalloc( (void**)&global_row_offsets_dev_ptr, ((numberOfRows+1)*treeSize) * sizeof(int) );
    cudaMalloc( (void**)&global_columns_dev_ptr, (numberOfEdgesPerGraph*treeSize) * sizeof(int) );
    cudaMalloc( (void**)&global_values_dev_ptr, (numberOfEdgesPerGraph*treeSize) * sizeof(int) );
    cudaMalloc( (void**)&global_degrees_dev_ptr, (numberOfRows*treeSize) * sizeof(int) );
    cudaMalloc( (void**)&global_paths_ptr, (max_dfs_depth*treeSize) * sizeof(int) );
    cudaMalloc( (void**)&global_remaining_vertices_ptr, (numberOfRows*treeSize) * sizeof(int) );

    cudaMalloc( (void**)&global_column_buffer, numberOfEdgesPerGraph * deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_value_buffer, numberOfEdgesPerGraph * deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_vertex_buffer, numberOfRows * deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_offsets_buffer, (numberOfRows+1) * deepestLevelSize * sizeof(int) );


    cudaMalloc( (void**)&global_paths_length, treeSize * sizeof(int) );
    cudaMalloc( (void**)&global_remaining_vertices_size_dev_ptr, treeSize * sizeof(int) );
    cudaMalloc( (void**)&global_nonpendant_path_bool_dev_ptr, deepestLevelSize * sizeof(int) );
    cudaMalloc( (void**)&global_nonpendant_child_dev_ptr, deepestLevelSize * sizeof(int) );

    cudaMalloc( (void**)&global_edges_left_to_cover_count, treeSize * sizeof(int) );

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    CopyGraphToDevice(g,
                    global_row_offsets_dev_ptr,
                    global_columns_dev_ptr,
                    global_values_dev_ptr,
                    global_degrees_dev_ptr,
                    numberOfEdgesPerGraph,
                    global_edges_left_to_cover_count,
                    global_remaining_vertices_ptr,
                    global_remaining_vertices_size_dev_ptr,
                    verticesRemainingInGraph);

    long long levelOffset = 0;
    long long levelUpperBound;
    int numberOfBlocksForOneThreadPerLeaf;
    numberOfLevels = 1;
    bool pendantNodeExists = true;

    int * pendantBools = new int[deepestLevelSize];
    int * pendantChildrenOfLevel = new int[deepestLevelSize];

    // Determine temporary device storage requirements
    int     *global_columns_tree = NULL;
    // Determine temporary device storage requirements
    int     *global_values_tree = NULL;

    for (int level = 0; level < numberOfLevels; ++level){
        levelUpperBound = CalculateLevelUpperBound(level);
        numberOfBlocksForOneThreadPerLeaf = ((levelUpperBound - levelOffset) + threadsPerBlock - 1) / threadsPerBlock;
        // 1 thread per leaf
        std::cout << "Calling DFS - level " << level << std::endl;
        // 1 block per leaf; tries tPB random paths in G
        // Hence threadsPerBlock*4,
        // Each thread checks it's path's pendant status
        // These booleans are reduced in shared memory
        // Hence + threadsPerBlock
        while(pendantNodeExists){
            std::cout << "pendantNodeExists - true " << std::endl;

            // Assumes all edges are turned on.  We need to compress a graph
            // after processing the edges of pendant paths
            int sharedMemorySize = threadsPerBlock*4 + threadsPerBlock;
            ParallelDFSRandom<<<levelUpperBound-levelOffset,threadsPerBlock,sharedMemorySize*sizeof(int)>>>
                                (levelOffset,
                                levelUpperBound,
                                numberOfRows,
                                numberOfEdgesPerGraph,
                                global_row_offsets_dev_ptr,
                                global_columns_dev_ptr,
                                global_remaining_vertices_ptr,
                                global_remaining_vertices_size_dev_ptr,
                                global_degrees_dev_ptr,
                                global_paths_ptr,
                                global_nonpendant_path_bool_dev_ptr,
                                global_nonpendant_child_dev_ptr);
            
            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);

            pendantNodeExists = false;
            cudaMemcpy(pendantBools, global_nonpendant_path_bool_dev_ptr, deepestLevelSize*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(pendantChildrenOfLevel, global_nonpendant_child_dev_ptr, deepestLevelSize*sizeof(int), cudaMemcpyDeviceToHost);
            

            for (int node = levelOffset; node < levelUpperBound; ++node){
                // global_nonpendant_path_bool_dev_ptr was defined as an OR of 
                // 0) path[0] == path[2]
                // 1) path[1] == path[3]
                std::cout << "node " << node << std::endl;
                std::cout << "global_nonpendant_path_bool_dev_ptr[node] " << pendantBools[node] << std::endl;

                std::cout << "!global_nonpendant_path_bool_dev_ptr[node] " << !pendantBools[node] << std::endl;

                if (!pendantBools[node]){
                    std::cout << "node " << node << " is pendant" << std::endl;

                    pendantNodeExists = true;
                    pendantChild = pendantChildrenOfLevel[node];

                    pendantChildren[node].push_back(pendantChild);
                    std::cout << "node " << node << "'s pendantChild " << pendantChild << " was pushed" << std::endl;
                }
            }
            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);
            // Each node assigned a block,  outgoing and incoming edges of child 
            // from pendant path processed at thread level
            // Block immediately returns if nonpendant path
            ParallelProcessPendantEdges<<<levelUpperBound-levelOffset,threadsPerBlock>>>
                            (levelOffset,
                            levelUpperBound,
                            numberOfRows,
                            numberOfEdgesPerGraph,
                            global_row_offsets_dev_ptr,
                            global_columns_dev_ptr,
                            global_values_dev_ptr,
                            global_remaining_vertices_ptr,
                            global_remaining_vertices_size_dev_ptr,
                            global_degrees_dev_ptr,
                            global_paths_ptr,
                            global_nonpendant_path_bool_dev_ptr);
            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);
          
            // Create pointer that starts at beginning of level
            // Leaves are indexed from 0; so I need to add the offset
            // of the leaf from the left of the tree * (numberOfRows+1) so the 
            // sorting operation works on an entire level.
            // global_offsets_buffer = &global_row_offsets_dev_ptr[levelOffset*(numberOfRows+1)];
            ParallelCreateLevelAwareRowOffsets<<<levelUpperBound-levelOffset,threadsPerBlock>>>
                                                (levelOffset,
                                                levelUpperBound,
                                                numberOfRows,
                                                numberOfEdgesPerGraph,
                                                global_row_offsets_dev_ptr,
                                                global_offsets_buffer);

            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);
            
            global_columns_tree = &global_columns_dev_ptr[levelOffset*numberOfEdgesPerGraph];
            global_values_tree = &global_values_dev_ptr[levelOffset*numberOfEdgesPerGraph];

            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int> d_keys(global_columns_tree, global_column_buffer);
            cub::DoubleBuffer<int> d_values(global_values_tree, global_value_buffer);

            // Determine temporary device storage requirements
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
            int num_items = (levelUpperBound-levelOffset)*numberOfEdgesPerGraph;
            int num_segments = (levelUpperBound-levelOffset)*(numberOfRows+1);

            cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values,
                num_items, num_segments, global_offsets_buffer, global_offsets_buffer + 1);

            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);

            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);

            // Run sorting operation
            cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values,
                num_items, num_segments, global_offsets_buffer, global_offsets_buffer + 1);

            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);

            int * printAlt = d_values.Alternate();
            std::cout << "Unsorted" << std::endl;
            for (int i = 0; i < (levelUpperBound-levelOffset)*numberOfEdgesPerGraph; ++i){
                std::cout << printAlt[i] << " ";
            }
            std::cout << std::endl;

            int * printCurr = d_values.Current();

            std::cout << "Sorted" << std::endl;
            for (int i = 0; i < (levelUpperBound-levelOffset)*numberOfEdgesPerGraph; ++i){
                std::cout << printCurr[i] << " ";
            }
            std::cout << std::endl;
        }
        
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
    }

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
    //cudaFree( global_vertices_remaining );
    //cudaFree( global_vertices_remaining_count );
    //cudaFree( global_outgoing_edge_vertices );
    //cudaFree( global_outgoing_edge_vertices_count );
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
                        int verticesRemainingInGraph){

    int * new_degrees_ptr = thrust::raw_pointer_cast(g.GetNewDegRef().data());
    int * vertices_remaining_ptr = thrust::raw_pointer_cast(g.GetRemainingVertices().data());
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