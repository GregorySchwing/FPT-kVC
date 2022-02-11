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

__host__ __device__ inline int32_t Pack(int16_t a, int16_t b)
{
   return (int32_t)((((uint32_t)a)<<16)+(uint32_t)b);
}

__host__ __device__ inline int16_t UnpackA(int32_t x)
{
   return (int16_t)(((uint32_t)x)>>16);
}

__host__ __device__ inline int16_t UnpackB(int32_t x)
{
   return (int16_t)(((uint32_t)x)&0xffff);
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

    int row = threadIdx.x + blockDim.x * blockIdx.x;

    inner_array_t *C_ref = new inner_array_t[numberOfRows];

    for (int iter = row; iter < numberOfRows; iter += blockDim.x){

        C_ref[iter][0] = 0;
        C_ref[iter][1] = 0;

        int beginIndex = old_row_offsets_dev[iter];
        int endIndex = old_row_offsets_dev[iter+1];

        for (int i = beginIndex; i < endIndex; ++i){
            ++C_ref[iter][old_values_dev[i]];
        }

        // This is  [old degree - new degree , new degree]
        for (int i = 1; i < 2; ++i){
            C_ref[iter][i] = C_ref[iter][i] + C_ref[iter][i-1];
        }

        /* C_ref[A_row_indices[i]]]-1 , because the values of C_ref are from [1, n] -> [0,n) */
        for (int i = endIndex-1; i >= beginIndex; --i){
            if (old_values_dev[i]){
                new_columns_dev[new_row_offsets_dev[iter] - C_ref[iter][0] + C_ref[iter][1]-1] = old_columns_dev[i];
                new_values_dev[new_row_offsets_dev[iter] - C_ref[iter][0] + C_ref[iter][1]-1] = old_values_dev[i];
                --C_ref[iter][old_values_dev[i]];
            }
        }
    }
    delete[] C_ref;
}

void CallInduceSubgraph(Graph & g, 
                    int * new_row_offs_dev,
                    int * new_cols_dev,
                    int * new_vals_dev,
                    int * new_degrees_dev,
                    int * new_row_offs_host,
                    int * new_cols_host,
                    int * new_vals_host){

    int numberOfRows = g.GetVertexCount(); 
    int numberOfEdgesPerGraph = g.GetEdgesLeftToCover();  // size M
    int condensedData = g.GetVertexCount(); // size N

    int condensedData_plus1 = condensedData + 1; // size N + 1
    long long sizeOfSingleGraph = numberOfEdgesPerGraph*2 + 2*condensedData + condensedData_plus1;
    long long totalMem = sizeOfSingleGraph * sizeof(int);

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
    //#ifndef NDEBUG
    //do 
    //{
    //    std::cout << '\n' << "Press enter to continue...; ctrl-c to terminate";
    //} while (std::cin.get() != '\n');
    //#endif

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    CopyGraphToDeviceAndInduce(g,
                      numberOfEdgesPerGraph,
                      new_row_offs_dev,
                      new_cols_dev,
                      new_vals_dev,
                      new_degrees_dev);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    CopyGraphFromDevice(g,
                        numberOfEdgesPerGraph,
                        new_row_offs_dev,
                        new_cols_dev,
                        new_row_offs_host,
                        new_cols_host);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
}

void CallMIS(   Graph & g,
                int * new_row_offs_dev,
                int * new_cols_dev,
                int * new_vals_dev,
                int * new_degrees_dev,
                int * triangle_remaining_boolean
            ){
    int remainingTrianglesCount = 1;
    while(remainingTrianglesCount){ 
        
    }
}

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
                        int * triangle_row_offsets_array_dev){

    cudaMemcpy(&new_row_offs_dev[0], &new_row_offs_host[0], (numberOfRows+1) * sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(&new_cols_dev[0], &new_cols_host[0], numberOfEdgesPerGraph * sizeof(int) , cudaMemcpyHostToDevice);
    
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(triangle_row_offsets_array_dev),  0, size_t(numberOfRows+1));
    
    // Updated final
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(triangle_counter_dev),  0, size_t(numberOfRows));

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    int oneThreadPerNode = (numberOfRows + threadsPerBlock - 1) / threadsPerBlock;
    CountTriangleKernel<<<oneThreadPerNode,threadsPerBlock>>>(  numberOfRows,
                                                                new_row_offs_dev,
                                                                new_cols_dev,
                                                                triangle_counter_dev
                                                            );
    
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    CalculateNewRowOffsets(numberOfRows,
                            triangle_row_offsets_array_dev,
                            triangle_counter_dev); 
      
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);  

    cudaMemcpy(&triangle_counter_host[0], &triangle_counter_dev[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);

    cudaMemcpy(numberOfTriangles_host, &triangle_row_offsets_array_dev[numberOfRows], 1 * sizeof(int) , cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

}

void CallSaveTriangles( int numberOfRows,
                        int numberOfTriangles,
                        int * new_row_offs_dev,
                        int * new_cols_dev,
                        int * triangle_row_offsets_array_host,
                        int * triangle_row_offsets_array_dev,
                        int * triangle_candidates_a_host,
                        int * triangle_candidates_b_host,
                        int * triangle_candidates_a_dev, 
                        int * triangle_candidates_b_dev){
    int oneThreadPerNode = (numberOfRows + threadsPerBlock - 1) / threadsPerBlock;
    SaveTrianglesKernel<<<oneThreadPerNode,threadsPerBlock>>>(  numberOfRows,
                                                                new_row_offs_dev,
                                                                new_cols_dev,
                                                                triangle_row_offsets_array_dev,
                                                                triangle_candidates_a_dev,
                                                                triangle_candidates_b_dev
                                                            );
    
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    cudaMemcpy(&triangle_row_offsets_array_host[0], &triangle_row_offsets_array_dev[0], (numberOfRows+1) * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&triangle_candidates_a_host[0], &triangle_candidates_a_dev[0], numberOfTriangles * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&triangle_candidates_b_host[0], &triangle_candidates_b_dev[0], numberOfTriangles * sizeof(int) , cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    for (int i = 0; i < numberOfRows; ++i){
        std::cout << "Node " << i << "'s candidate triangles:" << std::endl;
        int LB = triangle_row_offsets_array_host[i];
        int UB = triangle_row_offsets_array_host[i+1];
        for (int j = triangle_row_offsets_array_host[i]; j < triangle_row_offsets_array_host[i+1]; ++j){
            int a = triangle_candidates_a_host[j];
            int b = triangle_candidates_b_host[j];
            std::cout << "(" << a << ",  " << b << ") " << std::endl;
        }
        std::cout << std::endl;
    }
}


void CallDisjointSetTriangles(
    int numberOfRows,
    int * new_row_offs_dev,
    int * new_cols_dev,
    int * triangle_row_offsets_array_dev,
    int * triangle_counter_host,
    int * triangle_counter_dev,
    int * triangle_candidates_a_dev,
    int * triangle_candidates_b_dev){

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    int * L_dev;
    int * conflictsRemain_dev;
    int zero = 0;
    int * conflictsRemain;
    // To avoid warnings
    conflictsRemain = &zero;

    cudaMalloc( (void**)&L_dev, numberOfRows * sizeof(int) );
    cudaMalloc( (void**)&conflictsRemain_dev, 1 * sizeof(int) );

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    cuMemsetD32(reinterpret_cast<CUdeviceptr>(L_dev),  0, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(conflictsRemain_dev),  0, size_t(1));

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    int oneThreadPerNode = (numberOfRows + threadsPerBlock - 1) / threadsPerBlock;
    int * L_host = new int[numberOfRows];

    CheckForConficts<<<oneThreadPerNode,threadsPerBlock>>>(numberOfRows,
                                                            triangle_counter_dev,
                                                            conflictsRemain_dev);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    cudaMemcpy(conflictsRemain, conflictsRemain_dev, 1 * sizeof(int) , cudaMemcpyDeviceToHost);

    while(*conflictsRemain){
        IdentifyMaximumConflictTriangles<<<oneThreadPerNode,threadsPerBlock>>>( numberOfRows,
                                                                                new_row_offs_dev,
                                                                                new_cols_dev,
                                                                                triangle_counter_dev,
                                                                                L_dev
                                                                            );

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        cudaMemcpy(L_host, L_dev, numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        std::cout << "Max L" << std::endl;
        for (int i = 0; i < numberOfRows; ++i){
            std::cout << L_host[i] << " ";
        }
        std::cout << std::endl;

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        TurnOffMaximumEdgeOfConflictTriangles<<<oneThreadPerNode,threadsPerBlock>>>(  numberOfRows,
                                                                                triangle_row_offsets_array_dev,
                                                                                triangle_candidates_a_dev,
                                                                                triangle_candidates_b_dev,
                                                                                triangle_counter_dev,
                                                                                L_dev
                                                                            );

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        CheckForConficts<<<oneThreadPerNode,threadsPerBlock>>>(numberOfRows,
                                                                triangle_counter_dev,
                                                                conflictsRemain_dev);

        cudaMemcpy(conflictsRemain, conflictsRemain_dev, 1 * sizeof(int) , cudaMemcpyDeviceToHost);
        cudaMemcpy(triangle_counter_host, triangle_counter_dev, numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        std::cout << "Number of Triangles" << std::endl;
        for (int i = 0; i < numberOfRows; ++i){
            std::cout << triangle_counter_host[i] << " ";
        }
        std::cout << std::endl;

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
        if(*conflictsRemain){
            cuMemsetD32(reinterpret_cast<CUdeviceptr>(L_dev),  0, size_t(numberOfRows));
            cuMemsetD32(reinterpret_cast<CUdeviceptr>(conflictsRemain_dev),  0, size_t(1));
        }
    }

    cudaFree( L_dev );
    cudaFree( conflictsRemain_dev );
}

__global__ void CountTriangleKernel(int numberOfRows,
                                    int * new_row_offs_dev,
                                    int * new_cols_dev,
                                    int * triangle_counter_dev){

    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= numberOfRows)
        return;
    for (int i = new_row_offs_dev[v]; i < new_row_offs_dev[v+1]; ++i){
        int currMiddle = new_cols_dev[i];
        for (int j = new_row_offs_dev[currMiddle]; j < new_row_offs_dev[currMiddle+1]; ++j){
            int currLast = new_cols_dev[j];
            if (v != currLast && currMiddle < currLast){
                for (int k = new_row_offs_dev[currLast]; k < new_row_offs_dev[currLast+1]; ++k){
                    int candidateClose = new_cols_dev[k];
                    if (v == candidateClose){
                        triangle_counter_dev[v] = triangle_counter_dev[v] + 1;
                    }
                }
            }
        }
    }
}


__global__ void SaveTrianglesKernel(int numberOfRows,
                                    int * new_row_offs_dev,
                                    int * new_cols_dev,
                                    int * triangle_row_offsets_array_dev,
                                    int * triangle_candidates_a_dev,
                                    int * triangle_candidates_b_dev){


    int v = threadIdx.x + blockDim.x * blockIdx.x;

    if (v >= numberOfRows)
        return;
    int vertexOffset = triangle_row_offsets_array_dev[v];
    int vertexOffsetEnd = triangle_row_offsets_array_dev[v+1];
    printf ("%d %d %d\n", v, vertexOffset, vertexOffsetEnd);
    int totalTriangles = vertexOffsetEnd-vertexOffset;
    int triangleCounter = 0;
    int a;
    int b;
    for (int i = new_row_offs_dev[v]; i < new_row_offs_dev[v+1]; ++i){
        int currMiddle = new_cols_dev[i];
        for (int j = new_row_offs_dev[currMiddle]; j < new_row_offs_dev[currMiddle+1]; ++j){
            int currLast = new_cols_dev[j];
            if (v != currLast && currMiddle < currLast){
                for (int k = new_row_offs_dev[currLast]; k < new_row_offs_dev[currLast+1]; ++k){
                    int candidateClose = new_cols_dev[k];
                    if (v == candidateClose){
                        triangle_candidates_a_dev[vertexOffset + triangleCounter] = currMiddle;
                        triangle_candidates_b_dev[vertexOffset + triangleCounter] = currLast;
                        printf("Triangle %d (%d %d)\n", vertexOffset + triangleCounter, currMiddle, currLast);
                        ++triangleCounter;
                        if (triangleCounter == totalTriangles)
                            return;
                    }
                }
            }
        }
    }
}

__global__ void CalculateConflictDegree(int numberOfRows,
                                        int * triangle_row_offsets_array_dev,
                                        int * triangle_counter_dev,
                                        int * triangle_candidates_a_dev,
                                        int * triangle_candidates_b_dev){

    int v = threadIdx.x + blockDim.x * blockIdx.x;

    if (v >= numberOfRows)
        return;
    int a;
    int b;
    printf("Use v %d with rowofftri\n", v);
    printf("triangle_row_offsets_array_dev[%d] = %d\n", v,triangle_row_offsets_array_dev[v]);
    printf("triangle_row_offsets_array_dev[%d] = %d\n", v+1,triangle_row_offsets_array_dev[v+1]);
    for (int i = triangle_row_offsets_array_dev[v]; i < triangle_row_offsets_array_dev[v+1]; ++i){
        printf("about to Dereference a candidate triangle %d\n",i);
        a = triangle_candidates_a_dev[i];
        b = triangle_candidates_b_dev[i];
        printf("finished to Dereferencing a candidate triangle %d\n",i);

        //a = (int)vp.yz[0];
        //b = (int)vp.yz[1];

        printf("%d %d\n", a,b);
        atomicAdd(&triangle_counter_dev[a], 1);
        atomicAdd(&triangle_counter_dev[b], 1);
    }
}

__global__ void CheckForConficts(int numberOfRows,
                                int * triangle_counter_dev,
                                int * conflictsRemain){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= numberOfRows)
        return;
    if(triangle_counter_dev[v] > 1) 
        *conflictsRemain = 1;
}

__global__ void CalculateConflictDegreeNeighborhoodSum(int numberOfRows,
                                                        int * new_row_offs_dev,
                                                        int * new_cols_dev,
                                                        int * triangle_counter_dev,
                                                        int * conflictDegreeNeighborhoodSum_dev){


    int v = threadIdx.x + blockDim.x * blockIdx.x;
    int neighborhoodSum = 0;
    if (v >= numberOfRows)
        return;
    for (int i = new_row_offs_dev[v]; i < new_row_offs_dev[v+1]; ++i){
        neighborhoodSum += triangle_counter_dev[new_cols_dev[i]];
    }
    conflictDegreeNeighborhoodSum_dev[v] = neighborhoodSum;
}

__global__ void IdentifyMaximumConflictTriangles(int numberOfRows,
                                                int * new_row_offs_dev,
                                                int * new_cols_dev,
                                                int * triangle_counter_dev,
                                                int * L_dev){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= numberOfRows)
        return;
    // If I have no triangles, just return.
    if (triangle_counter_dev[v] == 0)
        return;
    int myCounter = triangle_counter_dev[v];
    int myNeighborCounter;
    int neighbor;
    int turnMyselfOff = true;
    for (int i = new_row_offs_dev[v]; i < new_row_offs_dev[v+1]; ++i){
        neighbor = new_cols_dev[i];
        myNeighborCounter = triangle_counter_dev[neighbor];
        // Assume true and only set false if find a larger neighbor
        // Either conflict sum or hash if conflict sums are equal
        if (myNeighborCounter < myCounter){
            turnMyselfOff = false;
        } else if (myNeighborCounter == myCounter && h(v) < h(neighbor)){
            turnMyselfOff = false;
        }
    }
    L_dev[v] = turnMyselfOff;
}

__device__ inline unsigned int h(unsigned int v){
    unsigned int x;
    x = ((v >> 16)^v)*0x45d9f3b;
    x = ((v >> 16)^x)*0x45d9f3b;
    x = ((v >> 16)^x);
    return x;
}

__global__ void TurnOffMaximumEdgeOfConflictTriangles(int numberOfRows,
                                                int * triangle_row_offsets_array_dev,
                                                int * triangle_candidates_a_dev,
                                                int * triangle_candidates_b_dev,
                                                int * triangle_counter_dev,
                                                int * L_dev){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= numberOfRows || !L_dev[v])
        return;

    int i = triangle_row_offsets_array_dev[v];
    int a = triangle_candidates_a_dev[i];   
    int b = triangle_candidates_b_dev[i];
    int maxA = a;
    int maxB = b;
    int edgeSum = triangle_counter_dev[a] + triangle_counter_dev[b];
    int maxEdgeSum = edgeSum;
    i = i + 1;
    for (; i < triangle_row_offsets_array_dev[v+1]; ++i){
        a = triangle_candidates_a_dev[i];        
        b = triangle_candidates_b_dev[i];
        edgeSum = triangle_counter_dev[a] + triangle_counter_dev[b];
        if(edgeSum > maxEdgeSum){
            maxA = a;
            maxB = b;
        }
    }
    printf("Turning off triangle %d %d %d\n", v, maxA, maxB);
    triangle_counter_dev[v] = triangle_counter_dev[v] - 1;
    triangle_counter_dev[maxA] = triangle_counter_dev[maxA] - 1;
    triangle_counter_dev[maxB] = triangle_counter_dev[maxB] - 1;
}

__global__ void DisjointSetTriangleKernel(int numberOfRows,
                                    int * new_row_offs_dev,
                                    int * new_cols_dev,
                                    int * triangle_counter_dev,
                                    int * triangle_reduction_array){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= numberOfRows)
        return;
    for (int i = new_row_offs_dev[v]; i < new_row_offs_dev[v+1]; ++i){
        int currMiddle = new_cols_dev[i];
        if (v < currMiddle){
            for (int j = new_row_offs_dev[currMiddle]; j < new_row_offs_dev[currMiddle+1]; ++j){
                int currLast = new_cols_dev[j];
                if (currMiddle < currLast){
                    for (int k = new_row_offs_dev[currLast]; k < new_row_offs_dev[currLast+1]; ++k){
                        int candidateClose = new_cols_dev[k];
                        if (v == candidateClose){
                            triangle_counter_dev[v] = triangle_counter_dev[v] + 1;
                        }
                    }
                }
            }
        }
    }
}

void SSSPAndBuildDepthCSR(Graph & g, 
                     int root,
                     int * new_row_offs_host,
                     int * new_cols_host,
                    int * Depth_CSR_host,
                    int * new_row_offs_dev,
                    int * new_cols_dev){

    // SSSP
    // Weights = size N
    int * global_W;
    // Mask = size M
    int * global_M;
    // Cost = size M
    int * global_C;
    // Update = size M
    int * global_U;
    // Intermediate = size M
    int * global_Pred;
    // Update = size M
    int * global_U_Pred;


    int numberOfRows = g.GetVertexCount(); 
    int numberOfEdgesPerGraph = g.GetEdgesLeftToCover();  // size M

    // SSSP
    // Can be removed for unweighted-graphs
    cudaMalloc( (void**)&global_W, numberOfEdgesPerGraph * sizeof(int) );
    cudaMalloc( (void**)&global_M, numberOfRows * sizeof(int) );
    // Intermediate
    cudaMalloc( (void**)&global_C, numberOfRows * sizeof(int) );
    // Updated final
    cudaMalloc( (void**)&global_U, numberOfRows * sizeof(int) );
    // Intermediate
    cudaMalloc( (void**)&global_Pred, numberOfRows * sizeof(int) );
    // Updated final
    cudaMalloc( (void**)&global_U_Pred, numberOfRows * sizeof(int) );

    // Reset SSSP vectors
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_W),  1, size_t(numberOfEdgesPerGraph));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_M),  0, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_C),  INT_MAX, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_U),  INT_MAX, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_U_Pred),  INT_MAX, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_Pred),  INT_MAX, size_t(numberOfRows));
    
    
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    // Root should optimally be a degree 1 vertex.
    // We never color the root, so by starting the sssp from
    // degree 1 vertices, this isn't a problem.
    // Root's neighbot also must have color cardinality < k.
    // Finally, the next root should be chosen at a maximal 
    // depth from the previous root.
    // The SSSP and Color algorithm ends when either the entire graph is colored
    // or no such vertices remain.
    PerformSSSP(numberOfRows,
                root,
                new_row_offs_dev,
                new_cols_dev,
                global_W,
                global_M,
                global_C,
                global_U,
                global_Pred,
                global_U_Pred);
    
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);


}
/*
void CreateDepthCSR(numberOfRows,
                    global_U,
                    ){
    // Declare, allocate, and initialize device-accessible pointers for sorting data
    int  num_items ;          // e.g., 7
    int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
    int  *d_keys_out;        // e.g., [        ...        ]
    int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
    int  *d_values_out;      // e.g., [        ...        ]

    thrust::device_vector<int> colors(g.GetNumberOfRows());
    // initialize X to 0,1,2,3, ....
    thrust::sequence(colors.begin(), colors.end());
    global_colors = thrust::raw_pointer_cast(colors.data());

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // d_keys_out            <-- [0, 3, 5, 6, 7, 8, 9]
    // d_values_out          <-- [5, 4, 3, 1, 2, 0, 6]
}

void ColorGraph(){



    int * global_levels; // size N, will contatin BFS level of nth node
    int * global_middle_vertex;
    int * global_colors; // size N, will contatin color of nth node
    int * global_color_card;   // size N, will contatin size of color set
    int * global_color_finished; // size N, contatin boolean of color finished, sometimes cardinality 1,2,3 is finished.
    int * global_finished_card_reduced;
    int * finished_gpu;
    int * nextroot_gpu;

    int numberOfRows = g.GetVertexCount(); 
    int numberOfEdgesPerGraph = g.GetEdgesLeftToCover();  // size M

    // Malloc'ed by thrust
    //cudaMalloc( (void**)&global_colors, numberOfRows * sizeof(int) );
    // middle vertex flag can be removed if colors with
    // cardinality of size 3 that are non-cycles
    // are reset to original colors every time.
    // This might be worth experimenting with for a low-memory version.
    cudaMalloc( (void**)&global_middle_vertex, numberOfRows * sizeof(int) );
    cudaMalloc( (void**)&global_color_card, numberOfRows * sizeof(int) );
    cudaMalloc( (void**)&global_color_finished, numberOfRows * sizeof(int) );
    cudaMalloc( (void**)&finished_gpu, 1 * sizeof(int) );
    cudaMalloc( (void**)&nextroot_gpu, 1 * sizeof(int) );
    cudaMalloc( (void**)&global_U_Prev, numberOfRows * sizeof(int) );
    cudaMalloc( (void**)&global_finished_card_reduced, 1 * sizeof(int) );


    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_color_card),  1, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_middle_vertex),  0, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_color_finished),  0, size_t(numberOfRows));


    int k = 4;

    // allocate three device_vectors with 10 elements
    thrust::device_vector<int> colors(g.GetNumberOfRows());
    // initialize X to 0,1,2,3, ....
    thrust::sequence(colors.begin(), colors.end());
    global_colors = thrust::raw_pointer_cast(colors.data());


    int host_reduced;
    double host_percentage_finished = 0;
    double percentage_threshold = 0.80;
    int iteration_threshold = 1;

    int done = 0;
    int iteration = 0;
    bool graphEveryIteration = true;
    
    std::string name = "main";
    std::string filenameGraphPrefix = "SSSP_iter_";
    bool isDirected = false;
    DotWriter::RootGraph gVizWriter(isDirected, name);
    std::string subgraph1 = "graph";
    std::string subgraph2 = "SSSP";

    std::map<std::string, DotWriter::Node *> nodeMap;    
    std::map<std::string, DotWriter::Node *> predMap;    

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 655); // define the range

    int * new_colors_randomized = new int[numberOfRows];
    int * new_colors_mapper = new int[numberOfRows];

    for(int n=0; n<numberOfRows; ++n){
        new_colors_mapper[n] = distr(gen); // generate numbers
    }


        std::cout << "Iteration : " << iteration << std::endl;
        std::cout << "Root : " << root << std::endl;
        std::cout << "Percentage Done : " << host_percentage_finished << std::endl;

        // Loop, X = k to 0
        // 1 thread per vertex
        // only vertices with cost a multiple of X are active on first iteration.
        // if so, write my color to the color of my predecessor
        // don't worry about race conditions, 
        // if two or more nodes share a predecessor,
        // last to write wins.  
        //The kernel call ends after every iteration to obtain SMP sync.

        // Since this is single-source shortest path,
        // all paths must eventually merge.  
        // Once they merge, they won't diverge.*
        // *TODO: double check this for divergent equal length paths.
        // *If they can diverge, modify algorithm to prevent as it could cause
        // Two colors of cardinality k/2, instead of 1 with k and the other < k.

        // Therefore, any color with a cardinality of K is an
        // independent path of length K.  
        // Colors with cardinality < K are disregarded.
        // Worst case: No paths of length K exist. (Star graph of depth K-1)
        // Best case: The entire graph is colored (Star graph of depth K+1)
        PerformPathPartitioning(numberOfRows,
                                k,
                                root,
                                new_row_offs_dev,
                                new_cols_dev,
                                global_middle_vertex,
                                global_M,
                                global_U_Prev,
                                global_U,
                                global_U_Pred,
                                global_colors,
                                global_color_card,
                                global_color_finished,
                                global_finished_card_reduced,
                                finished_gpu,
                                nextroot_gpu);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        cudaMemcpy(&host_reduced, &global_finished_card_reduced[0], 1 * sizeof(int) , cudaMemcpyDeviceToHost);
        cudaMemcpy(&root, &nextroot_gpu[0], 1 * sizeof(int) , cudaMemcpyDeviceToHost);
        cudaMemcpy(&global_U_Prev[0], &global_U[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToDevice);
        cudaMemcpy(&new_color_finished[0], &global_color_finished[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);

        host_percentage_finished = ((double)host_reduced/(double)numberOfRows);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        done |= percentage_threshold < host_percentage_finished;
        done |= iteration_threshold < iteration;

        ++iteration;

        if (graphEveryIteration){
//            cudaMemcpy(&new_row_offs[0], &global_row_offsets_dev_ptr[0], (numberOfRows+1) * sizeof(int) , cudaMemcpyDeviceToHost);
//            cudaMemcpy(&new_cols[0], &global_columns_dev_ptr[0], numberOfEdgesPerGraph * sizeof(int) , cudaMemcpyDeviceToHost);
            cudaMemcpy(&new_colors[0], &global_colors[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);
            cudaMemcpy(&host_U[0], &global_U[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);
            cudaMemcpy(&new_Pred[0], &global_U_Pred[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);
            
            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);

            for(int n=0; n<numberOfRows; ++n){
                new_colors_randomized[n] = new_colors_mapper[new_colors[n]]; // generate numbers
            }
    
            int maxdepth = 0;
            for (int i = 0; i < numberOfRows; ++i){
                if (host_U[i] > maxdepth && host_U[i] != INT_MAX){
                    maxdepth = host_U[i];
                }
            }
            int w = 0;
            int c = 0;
            predMap.clear();
            nodeMap.clear();
            DotWriter::Subgraph * graph = gVizWriter.AddSubgraph(subgraph1);
            DotWriter::Subgraph * pred = gVizWriter.AddSubgraph(subgraph2);

            // Since the graph doesnt grow uniformly, it is too difficult to only copy the new parts..
            for (int i = 0; i < numberOfRows; ++i){
                std::string node1Name = std::to_string(i);
                std::map<std::string, DotWriter::Node *>::const_iterator nodeIt1 = nodeMap.find(node1Name);
                if(nodeIt1 == nodeMap.end()) {
                    nodeMap[node1Name] = graph->AddNode(node1Name);
                    if(new_color_finished[new_colors[i]]){
                        nodeMap[node1Name]->GetAttributes().SetColor(DotWriter::Color::e(new_colors_randomized[i]));
                        nodeMap[node1Name]->GetAttributes().SetFillColor(DotWriter::Color::e(new_colors_randomized[i]));
                        nodeMap[node1Name]->GetAttributes().SetStyle("filled");
                    }
                }
                for (int j = new_row_offs_host[i]; j < new_row_offs_host[i+1]; ++j){
                    if (i < new_cols_host[j]){
                        std::string node2Name = std::to_string(new_cols_host[j]);
                        std::map<std::string, DotWriter::Node *>::const_iterator nodeIt2 = nodeMap.find(node2Name);
                        if(nodeIt2 == nodeMap.end()) {
                            nodeMap[node2Name] = graph->AddNode(node2Name);
                            if(new_color_finished[new_colors[new_cols_host[j]]]){
                                nodeMap[node2Name]->GetAttributes().SetColor(DotWriter::Color::e(new_colors_randomized[new_cols_host[j]]));
                                nodeMap[node2Name]->GetAttributes().SetFillColor(DotWriter::Color::e(new_colors_randomized[new_cols_host[j]]));
                                nodeMap[node2Name]->GetAttributes().SetStyle("filled");
                            }
                        }  
                        //graph->AddEdge(nodeMap[node1Name], nodeMap[node2Name], std::to_string(host_levels[i]));
                        graph->AddEdge(nodeMap[node1Name], nodeMap[node2Name]); 
         
                    }
                }
            }

            //pred->clear();
            for (int depth = 0; depth <= maxdepth; ++depth){
                for (int i = 0; i < numberOfRows; ++i){
                    if (host_U[i] == depth){
                        w = new_Pred[i];
                        c = new_colors[i];
                        std::string node1Name = std::to_string(i);
                        std::map<std::string, DotWriter::Node *>::const_iterator nodeIt1 = predMap.find(node1Name);
                        if(nodeIt1 == predMap.end()) {
                            predMap[node1Name] = pred->AddNode(node1Name);
                            if(new_color_finished[new_colors[i]]){
                                predMap[node1Name]->GetAttributes().SetColor(DotWriter::Color::e(new_colors_randomized[i]));
                                predMap[node1Name]->GetAttributes().SetFillColor(DotWriter::Color::e(new_colors_randomized[i]));
                                predMap[node1Name]->GetAttributes().SetStyle("filled");
                            }
                        }
                        std::string node2Name = std::to_string(w);
                        std::map<std::string, DotWriter::Node *>::const_iterator nodeIt2 = predMap.find(node2Name);
                        if(nodeIt2 == predMap.end()) {
                            predMap[node2Name] = pred->AddNode(node2Name);
                            if(new_color_finished[new_colors[w]]){
                                predMap[node2Name]->GetAttributes().SetColor(DotWriter::Color::e(new_colors_randomized[w]));
                                predMap[node2Name]->GetAttributes().SetFillColor(DotWriter::Color::e(new_colors_randomized[w]));
                                predMap[node2Name]->GetAttributes().SetStyle("filled");
                            }
                        }
                        pred->AddEdge(predMap[node1Name], predMap[node2Name], std::to_string(c)); 
                    }
                }
            }
            std::string iterFileName = filenameGraphPrefix+std::to_string(iteration)+".viz";
            gVizWriter.WriteToFile(iterFileName);
            gVizWriter.RemoveSubgraph(graph);
            gVizWriter.RemoveSubgraph(pred);
            //pred->~Subgraph();
            std::cout << "finished writing " << iterFileName << std::endl;
        }
    }


    //cudaMemcpy(&host_levels[0], &global_levels[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);
//    cudaMemcpy(&new_row_offs[0], &global_row_offsets_dev_ptr[0], (numberOfRows+1) * sizeof(int) , cudaMemcpyDeviceToHost);
//    cudaMemcpy(&new_cols[0], &global_columns_dev_ptr[0], numberOfEdgesPerGraph * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&new_colors[0], &global_colors[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_U[0], &global_U[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&new_Pred[0], &global_U_Pred[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);


    cudaFree( global_W );
    cudaFree( global_M );
    cudaFree( global_C );
    cudaFree( global_U );

    delete[] new_colors_randomized;
    delete[] new_colors_mapper;

    cudaDeviceSynchronize();


    //cudaFree( global_levels );
    cudaFree( finished_gpu );
    cudaFree( global_finished_card_reduced );
    //cudaFree( global_colors );
    cudaFree( global_color_card );
    cudaFree( global_color_finished );
    cudaFree( global_middle_vertex );
  
}
  */
void CopyGraphToDeviceAndInduce( Graph & g,
                        int numberOfEdgesPerGraph,
                        int * global_row_offsets_dev_ptr,
                        int * global_columns_dev_ptr,
                        int * global_values_dev_ptr,
                        int * global_degrees_dev_ptr){

    int * new_degrees_ptr = thrust::raw_pointer_cast(g.GetNewDegRef().data());

    // Degree CSR Data
    cudaMemcpy(global_degrees_dev_ptr, new_degrees_ptr, g.GetNumberOfRows() * sizeof(int),
                cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    // CSR vectors
    thrust::device_vector<int> old_row_offsets_dev = *(g.GetCSR().GetOldRowOffRef());
    thrust::device_vector<int> old_column_indices_dev = *(g.GetCSR().GetOldColRef());

    // SparseMatrix vectors
    thrust::device_vector<int> new_values_dev = g.GetCSR().GetNewValRef();
    // CSR pointers
    int * old_row_offsets_dev_ptr = thrust::raw_pointer_cast(old_row_offsets_dev.data());
    int * old_column_indices_dev_ptr = thrust::raw_pointer_cast(old_column_indices_dev.data());
    // SparseMatrix pointers
    int * new_values_dev_ptr = thrust::raw_pointer_cast(new_values_dev.data());

    // Currenly only sets the first graph in the cuda memory
    // Might as well be host code
    CalculateNewRowOffsets(g.GetNumberOfRows(),
                            global_row_offsets_dev_ptr,
                            global_degrees_dev_ptr); 
       
    
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    #ifndef NDEBUG
        std::cout << "NRO" << std::endl;
        int * new_row_offs = new int[g.GetNumberOfRows()+1];
        cudaMemcpy( &new_row_offs[0], &global_row_offsets_dev_ptr[0], (g.GetNumberOfRows()+1) * sizeof(int) , cudaMemcpyDeviceToHost);
        for (int i = 0; i < g.GetNumberOfRows()+1; ++i){
            printf("%d ",new_row_offs[i]);
        }
        printf("\n");
    #endif

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

void CopyGraphFromDevice(Graph & g,
                        int numberOfEdgesPerGraph,
                        int * global_row_offsets_dev_ptr,
                        int * global_columns_dev_ptr,
                        int * host_row_offsets,
                        int * host_columns){
    cudaMemcpy(&host_row_offsets[0], &global_row_offsets_dev_ptr[0], (g.GetNumberOfRows()+1) * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_columns[0], &global_columns_dev_ptr[0], g.GetEdgesLeftToCover() * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
}

void PerformBFS(int numberOfRows,
                int * global_levels,
                int * global_row_offsets_dev_ptr,
                int * global_columns_dev_ptr){

        int oneThreadPerNode = (numberOfRows + threadsPerBlock - 1) / threadsPerBlock;
        int curr = 0;
        int zero;
        int * finished = &zero;
        int * finished_gpu;
        cudaMalloc( (void**)&finished_gpu, 1 * sizeof(int) );
        cuMemsetD32(reinterpret_cast<CUdeviceptr>(finished_gpu),  0, size_t(1));
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
    
        do {
            cuMemsetD32(reinterpret_cast<CUdeviceptr>(finished_gpu),  1, size_t(1));
            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);
            launch_gpu_bfs_kernel<<<oneThreadPerNode,threadsPerBlock>>>(numberOfRows,
                                    curr++, 
                                    global_levels,
                                    global_row_offsets_dev_ptr,
                                    global_columns_dev_ptr,
                                    finished_gpu);
            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);
            cudaMemcpy(&finished[0], &finished_gpu[0], 1 * sizeof(int) , cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);
        } while (!(*finished));
    
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
}

void PerformSSSP(int numberOfRows,
                int root,
                int * global_row_offsets_dev_ptr,
                int * global_columns_dev_ptr,
                int * global_W,
                int * global_M,
                int * global_C,
                int * global_U,
                int * global_Pred,
                int * global_U_Pred){

        int oneThreadPerNode = (numberOfRows + threadsPerBlock - 1) / threadsPerBlock;
        int zero;
        int * finished = &zero;
        int * finished_gpu;

        cudaMalloc( (void**)&finished_gpu, 1 * sizeof(int) );
        cuMemsetD32(reinterpret_cast<CUdeviceptr>(&global_M[root]),  1, size_t(1));
        cuMemsetD32(reinterpret_cast<CUdeviceptr>(&global_C[root]),  0, size_t(1));
        cuMemsetD32(reinterpret_cast<CUdeviceptr>(&global_U[root]),  0, size_t(1));
        cuMemsetD32(reinterpret_cast<CUdeviceptr>(&global_Pred[root]),  0, size_t(1));

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
    
        do {
            cuMemsetD32(reinterpret_cast<CUdeviceptr>(finished_gpu),  1, size_t(1));
            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);

            launch_gpu_sssp_kernel_1<<<oneThreadPerNode,threadsPerBlock>>>(
                                    numberOfRows,
                                    global_row_offsets_dev_ptr,
                                    global_columns_dev_ptr,
                                    global_W,
                                    global_M,
                                    global_C,
                                    global_U,
                                    global_U_Pred);

            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);

            launch_gpu_sssp_kernel_2<<<oneThreadPerNode,threadsPerBlock>>>(
                                    numberOfRows,
                                    global_M,
                                    global_C,
                                    global_U,
                                    global_Pred,
                                    global_U_Pred);

            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);

            Sum(numberOfRows, global_M, finished_gpu);
            cudaMemcpy(&finished[0], &finished_gpu[0], 1 * sizeof(int) , cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);
        } while ((*finished));
    
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
}

void GlobalPathPartition(int numberOfRows,
                        int k,
                        int * global_row_offsets_dev_ptr,
                        int * global_columns_dev_ptr,
                        int * global_U,
                        int * global_colors,
                        int * global_colors_prev,
                        int * global_color_card,
                        int * global_color_finished,
                        int * global_finished_card_reduced,
                        int * finished_gpu){
    // Kernel to maximize the number of finished colors in a depth block of size 4
    // For each vertex in depth D, either delete an edge, add an edge, or both delete and add an edge
    
}

__global__ void launch_recolor_depth_block( int N,
                                            int k,
                                            int * M,
                                            int * U,
                                            int * colors,
                                            int * color_finished,
                                            int * middle_vertex){
    //int v = threadIdx.x + blockDim.x * blockIdx.x;
    
}



// Partitions path from root
// 1 to k
// k+1 to 2k
// ...
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
                            int * nextroot_gpu){

    int oneThreadPerNode = (numberOfRows + threadsPerBlock - 1) / threadsPerBlock;
    int maximizePathLength = true;

    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_finished_card_reduced),  0, size_t(1));

    // k-1 iterations to prevent claiming the root
    for (int iter = 0; iter < k-1; ++iter){
        launch_gpu_sssp_coloring_1<<<oneThreadPerNode,threadsPerBlock>>>(
                                                                        numberOfRows,
                                                                        k,
                                                                        iter,
                                                                        global_M,
                                                                        global_U,
                                                                        global_U_Pred,
                                                                        global_colors,
                                                                        global_color_finished,
                                                                        global_middle_vertex);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        // Loop until winner is largest cardinality color
        // Worst case number of loops is k
        if (maximizePathLength)
            MaximizePathLength(numberOfRows,
                                k,
                                iter,
                                finished_gpu,
                                global_M,
                                global_U,
                                global_U_Pred,
                                global_colors,
                                global_color_card,
                                global_color_finished,
                                global_middle_vertex);
        // Increment cardinality of winner
        launch_gpu_sssp_coloring_2<<<oneThreadPerNode,threadsPerBlock>>>(
                                                                        numberOfRows,
                                                                        k,
                                                                        iter,
                                                                        global_M,
                                                                        global_U,
                                                                        global_U_Pred,
                                                                        global_colors,
                                                                        global_color_card,
                                                                        global_color_finished,
                                                                        global_middle_vertex);
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
    }

    launch_gpu_color_finishing_kernel_1<<<oneThreadPerNode,threadsPerBlock>>>(
                                                                            numberOfRows,
                                                                            global_row_offsets_dev_ptr,
                                                                            global_columns_dev_ptr,
                                                                            global_colors,
                                                                            global_color_card,
                                                                            global_color_finished,
                                                                            global_middle_vertex);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    launch_gpu_color_finishing_kernel_2<<<oneThreadPerNode,threadsPerBlock>>>(
                                                                            numberOfRows,
                                                                            global_row_offsets_dev_ptr,
                                                                            global_columns_dev_ptr,
                                                                            global_colors,
                                                                            global_color_card,
                                                                            global_color_finished,
                                                                            global_middle_vertex);


    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    launch_gpu_color_finishing_kernel_2<<<oneThreadPerNode,threadsPerBlock>>>(
        numberOfRows,
        global_row_offsets_dev_ptr,
        global_columns_dev_ptr,
        global_colors,
        global_color_card,
        global_color_finished,
        global_middle_vertex);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    FindMaximumDistanceNonFinishedColor(numberOfRows,
                                        global_colors,
                                        global_M,
                                        global_color_finished,
                                        global_U_Prev,
                                        global_U,
                                        nextroot_gpu);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
       
    Sum(numberOfRows,
        global_color_finished,
        global_finished_card_reduced);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

}

__host__ void CalculateNewRowOffsets( int numberOfRows,
                                        int * global_row_offsets_dev_ptr,
                                        int * global_degrees_dev_ptr){
    // Declare, allocate, and initialize device-accessible pointers for input and output
    int  num_items = numberOfRows;      // e.g., 7
    int  *d_in = global_degrees_dev_ptr;        // e.g., [8, 6, 7, 5, 3, 0, 9]
    int  *d_out = &global_row_offsets_dev_ptr[1];         // e.g., [ ,  ,  ,  ,  ,  ,  ]
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // d_out s<-- [0, 8, 14, 21, 26, 29, 29]
    cudaFree(d_temp_storage);
}

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
                        int * global_middle_vertex){
    int zero = 0;
    int * finished = &zero;
    int oneThreadPerNode = (numberOfRows + threadsPerBlock - 1) / threadsPerBlock;

    do {
        cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_M),  0, size_t(numberOfRows));
        cuMemsetD32(reinterpret_cast<CUdeviceptr>(finished_gpu),  0, size_t(1));

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        launch_gpu_sssp_coloring_maximize<<<oneThreadPerNode,threadsPerBlock>>>(
                        numberOfRows,
                        k,
                        iter,
                        global_M,
                        global_U,
                        global_U_Pred,
                        global_colors,
                        global_color_card,
                        global_color_finished,
                        global_middle_vertex);

        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);

        Sum(numberOfRows, global_M, finished_gpu);
        cudaMemcpy(&finished[0], &finished_gpu[0], 1 * sizeof(int) , cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
    } while (*finished);
}

void FindMaximumDistanceNonFinishedColor(int numberOfRows,
                                        int * global_colors,
                                        int * global_M,
                                        int * global_color_finished,
                                        int * global_U_Prev,
                                        int * global_U,
                                        int * nextroot_gpu){

    int oneThreadPerNode = (numberOfRows + threadsPerBlock - 1) / threadsPerBlock;

    multiply_distance_by_finished_boolean<<<oneThreadPerNode,threadsPerBlock>>>(numberOfRows,
                                                                                global_M,
                                                                                global_colors,
                                                                                global_color_finished,
                                                                                global_U_Prev,
                                                                                global_U); 
                                                                                
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    GetMaxDist(numberOfRows,
                global_M,
                nextroot_gpu);

}

__global__ void multiply_distance_by_finished_boolean(int N,
                                                    int * M,
                                                    int * colors,
                                                    int * color_finished,
                                                    int * U_Prev,
                                                    int * U){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N)
        return;

    int vc = colors[v];
    M[v] = color_finished[vc]*U[v] + color_finished[vc]*U_Prev[v]; 
}

void GetMaxDist(int N,
                int * M,
                int * nextroot_gpu){
    // Declare, allocate, and initialize device-accessible pointers for input and output
    int  num_items = N;      // e.g., 7
    int  *d_in = M;          // e.g., [8, 6, 7, 5, 3, 0, 9]
    int  *d_out = nextroot_gpu;         // e.g., [-]
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run max-reduction
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // d_out <-- [9]
    cudaFree(d_temp_storage);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
}


__global__ void launch_gpu_bfs_kernel( int N, int curr, int *levels,
                                            int *nodes, int *edges, int * finished){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N)
        return;
    if (levels[v] == curr) {
        // iterate over neighbors
        int num_nbr = nodes[v+1] - nodes[v];
        int * nbrs = & edges[ nodes[v] ];
        for(int i = 0; i < num_nbr; i++) {
            int w = nbrs[i];
            if (levels[w] == INT_MAX) { // if not visited yet
                *finished = 0;
                levels[w] = curr + 1;
            }
        }
    }
}

// Processes colors of cardinality 3 and 4
// If a color with cardinality 3 is a cycle, finish the color
// else, mark the middle so we dont add a four vertex onto the middle.
__global__ void launch_gpu_color_finishing_kernel_1( int N,
                                                int * nodes,
                                                int * edges,
                                                int * colors,
                                                int * color_card,
                                                int * color_finished,
                                                int * middle_vertex){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N)
        return;
    int cv = colors[v];
    int myOwnColorEdges = 0;
    int middleVertex;
    // Guaruntees we need another vertex and v is not an internal vertex
    // for example, with path a - b - c; we guaruntee v is not b.
    if (color_card[cv] == 3 && !color_finished[cv]) {
        // iterate over neighbors
        int num_nbr = nodes[v+1] - nodes[v];
        int * nbrs = & edges[ nodes[v] ];
        for(int i = 0; i < num_nbr; i++) {
            int w = nbrs[i];
            int cw = colors[w];
            if (cw  == cv){
                ++myOwnColorEdges;
            }
        }
        middleVertex = myOwnColorEdges > 1;
        middle_vertex[v] = middleVertex;
    }
}

__global__ void launch_gpu_color_finishing_kernel_2( int N,
                                                int * nodes,
                                                int * edges,
                                                int * colors,
                                                int * color_card,
                                                int * color_finished,
                                                int * middle_vertex){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N)
        return;
    int cv = colors[v];
    int middleVertexCount = 0;
    int foundCycle;
    // Guaruntees we need another vertex and v is not an internal vertex
    // for example, with path a - b - c; we guaruntee v is not b.
    // middle_vertex[v] is required to prevent race conditions 
    // since only those with 
    if (middle_vertex[v] && color_card[cv] == 3 && !color_finished[cv]) {
        // iterate over neighbors
        int num_nbr = nodes[v+1] - nodes[v];
        int * nbrs = & edges[ nodes[v] ];
        for(int i = 0; i < num_nbr; i++) {
            int w = nbrs[i];
            int mvw = middle_vertex[w];
            int cw = colors[w];
            if (cw  == cv)
                middleVertexCount = middleVertexCount + mvw;
        }
        foundCycle = middleVertexCount > 1;
        color_finished[cv] = foundCycle;
    } else if (color_card[cv] == 4){
        color_finished[cv] = true;
    }
}

// The depth of all of vertex v's neighbors's u1, u2, ... 
// are either v's depth D, D-1, or D+1.
// By holding D constant, and requiring the depth of u1 be
// either D, D-1, or D+1; over three consecutive kernel calls,
// v's color is guarunteed to only grow, meaning another vertex
// w can't claim v at the same time v claims u.

// This way v can mark u1, and there is no chance v will also
// be marked.
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
                                                int * color_card){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N)
        return;
    if ((U[v] + iter) % k != 0 || U[v] == INT_MAX)
        return;
    
    int cv = colors[v];
    // If I didn't win
//    if(cv != cw)
    // Guaruntees we need another vertex and v is not an internal vertex
    // for example, with path a - b - c; we guaruntee v is not b.
    if (color_card[cv] < k && color_card[cv] % ((U[v] + iter) % k) == 0) {
        // iterate over neighbors
        int num_nbr = nodes[v+1] - nodes[v];
        int * nbrs = & edges[ nodes[v] ];
        for(int i = 0; i < num_nbr; i++) {
            int w = nbrs[i];
            int cw = colors[w];
            // My cardinality is larger than neighbor
            // Have to figure out how to prevent 
            if (color_card[cw] <= color_card[cv]) { 
                colors[w] = cv;
            }
        }
    }
}


__global__ void launch_gpu_sssp_kernel_1(   int N,      
                                            int * nodes,
                                            int * edges,
                                            int * W,
                                            int * M,
                                            int * C,
                                            int * U,
                                            int * U_Pred){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N || !M[v])
        return;
    else {
        M[v] = false;
        int num_nbr = nodes[v+1] - nodes[v];
        int * nbrs = & edges[ nodes[v] ];
        for(int i = 0; i < num_nbr; i++) {
            int w = nbrs[i];
            if (U[w] > C[v] + W[w]){
                U[w] = C[v] + W[w];
                U_Pred[w] = v;
            }
        }
    }
}

__global__ void launch_gpu_sssp_kernel_2(   int N,     
                                            int * M,
                                            int * C,
                                            int * U,
                                            int * Pred,
                                            int * U_Pred){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N) {
        return;
    } else {
        if (C[v] > U[v]){
            C[v] = U[v];
            Pred[v] = U_Pred[v];
            M[v] = true;
        }
        U[v] = C[v];
        U_Pred[v] = Pred[v];
    }
}

__global__ void launch_gpu_sssp_coloring_1(int N,
                                        int k,
                                        int iter,
                                        int * M,
                                        int * U,
                                        int * U_Pred,
                                        int * colors,
                                        int * color_finished,
                                        int * middle_vertex){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N)
        return;
    if ((U[v] + iter) % k != 0 || U[v] == INT_MAX)
        return;
    int w = U_Pred[v];
    int vc = colors[v];
    int wc = colors[w];

    // Race condition, but the kernel ends, so we get synchronization
    // it doesn't matter who wins, we need to initiate change in the graph.
    if (!color_finished[wc] && !color_finished[vc]){
        colors[w] = colors[v];
    }
}


__global__ void reset_partial_paths(int N,
                                    int * colors,
                                    int * color_card,
                                    int * color_finished,
                                    int * middle_vertex){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N)
        return;
    int vc = colors[v];
    if (color_finished[vc])
        return;
    colors[v] = v;
    color_card[v] = 1;
    middle_vertex[v] = 0;
}   

__global__ void calculate_percent_partitioned(int N,
                                                int * color_card,
                                                int * color_finished,
                                                int * finished_card_reduced){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N)
        return;
    if (color_finished[v])
        atomicAdd(finished_card_reduced, 1);
}

__global__ void launch_gpu_sssp_coloring_maximize(int N,
                                        int k,
                                        int iter,
                                        int * M,
                                        int * U,
                                        int * U_Pred,
                                        int * colors,
                                        int * color_card,
                                        int * color_finished,
                                        int * middle_vertex){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N)
        return;
    if ((U[v] + iter) % k != 0 || U[v] == INT_MAX)
        return;

    int vc = colors[v];
    int vcc = color_card[vc];
    
    int w = U_Pred[v];
    int wc = colors[w];
    int wcc = color_card[wc];
    // Still a race condition, of what color is assigned color[w]
    // so we need to keep calling this method until no marked vertices exist.
    if (vcc > wcc && !color_finished[vc] && !color_finished[wc]){
        colors[w] = colors[v];
        M[w] = 1;
    }
}

__global__ void launch_gpu_sssp_coloring_2(int N,
                                        int k,
                                        int iter,
                                        int * M,
                                        int * U,
                                        int * U_Pred,
                                        int * colors,
                                        int * color_card,
                                        int * color_finished,
                                        int * middle_vertex){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= N)
        return;
    if ((U[v] + iter) % k != 0 || U[v] == INT_MAX)
        return;

    int w = U_Pred[v];
    int wc = colors[w];
    int vc = colors[v];
    if(!color_finished[wc] && !color_finished[vc])
        color_card[wc] = color_card[wc] + 1;
}

void Sum(int expanded_size,
        int * expanded,
        int * reduced){
    // Declare, allocate, and initialize device-accessible pointers for input and output
    int  num_items = expanded_size;      // e.g., 7
    int  *d_in = expanded;          // e.g., [8, 6, 7, 5, 3, 0, 9]
    int  *d_out = reduced;         // e.g., [-]
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // d_out <-- [38]
    cudaFree(d_temp_storage);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
}
#endif
