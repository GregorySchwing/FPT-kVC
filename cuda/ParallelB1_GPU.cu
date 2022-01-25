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
        #ifndef NDEBUG
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
        #endif
    }
    delete[] C_ref;
}

void CallPopulateTree(Graph & g, int root){

    int * global_row_offsets_dev_ptr; // size N + 1
    int * global_degrees_dev_ptr; // size N, used for inducing the subgraph
    int * global_columns_dev_ptr; // size M
    int * global_values_dev_ptr; // on or off, size M
    int * global_levels; // size N, will contatin BFS level of nth node

    int expandedData = g.GetEdgesLeftToCover();  // size M
    int condensedData = g.GetVertexCount(); // size N

    int condensedData_plus1 = condensedData + 1; // size N + 1
    long long sizeOfSingleGraph = expandedData*2 + 2*condensedData + condensedData_plus1;
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
    #ifndef NDEBUG
    do 
    {
        std::cout << '\n' << "Press enter to continue...; ctrl-c to terminate";
    } while (std::cin.get() != '\n');
    #endif

// Vertices and Edges

    int numberOfRows = g.GetNumberOfRows();
    int numberOfEdgesPerGraph = g.GetEdgesLeftToCover(); 
    int verticesRemainingInGraphCount = g.GetRemainingVertices().size(); 

    // Vertex, Cols, Edge(on/off)
    cudaMalloc( (void**)&global_row_offsets_dev_ptr, (numberOfRows+1) * sizeof(int) );
    cudaMalloc( (void**)&global_columns_dev_ptr, numberOfEdgesPerGraph * sizeof(int) );
    cudaMalloc( (void**)&global_values_dev_ptr, numberOfEdgesPerGraph * sizeof(int) );
    cudaMalloc( (void**)&global_degrees_dev_ptr, (numberOfRows+1) * sizeof(int) );
    cudaMalloc( (void**)&global_levels, numberOfRows * sizeof(int) );

    // Set all levels value to root
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_levels),  root, size_t(numberOfRows));

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    CopyGraphToDevice(g,
                    numberOfEdgesPerGraph,
                    global_row_offsets_dev_ptr,
                    global_columns_dev_ptr,
                    global_values_dev_ptr,
                    global_degrees_dev_ptr);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    int oneThreadPerNode = (numberOfRows + threadsPerBlock - 1) / threadsPerBlock;
    int curr = 0;
    int finished = 0;
    do {
        finished = true;
        launch_gpu_bfs_kernel<<<threadsPerBlock,oneThreadPerNode>>>(numberOfRows,
                                curr++, 
                                global_levels,
                                global_row_offsets_dev_ptr,
                                global_columns_dev_ptr,
                                &finished);
    } while (!finished);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    cudaFree( global_row_offsets_dev_ptr );
    cudaFree( global_columns_dev_ptr );
    cudaFree( global_values_dev_ptr );
    cudaFree( global_degrees_dev_ptr );
    cudaFree( global_levels );
    cudaDeviceSynchronize();
}

void CopyGraphToDevice( Graph & g,
                        int numberOfEdgesPerGraph,
                        int * global_row_offsets_dev_ptr,
                        int * global_columns_dev_ptr,
                        int * global_values_dev_ptr,
                        int * global_degrees_dev_ptr){

    int * new_degrees_ptr = thrust::raw_pointer_cast(g.GetNewDegRef().data());
    std::cout << "remaining verts" << std::endl;
    for (auto & v : g.GetRemainingVerticesRef())
        std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "remaining verts size " << g.GetRemainingVerticesRef().size() << std::endl;
    
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

__host__ void CalculateNewRowOffsets( int numberOfRows,
                                        int * global_row_offsets_dev_ptr,
                                        int * global_degrees_dev_ptr){
    // Declare, allocate, and initialize device-accessible pointers for input and output
    int  num_items = numberOfRows+1;      // e.g., 7
    int  *d_in = global_row_offsets_dev_ptr;        // e.g., [8, 6, 7, 5, 3, 0, 9]
    int  *d_out = global_row_offsets_dev_ptr;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
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

__global__ void launch_gpu_bfs_kernel( int N, int curr, int *levels,
                                            int *nodes, int *edges, int * finished){
    int v = threadIdx.x;
    if (levels[v] == curr) {
        // iterate over neighbors
        int num_nbr = nodes[v+1] - nodes[v];
        int * nbrs = & edges[ nodes[v] ];
        for(int i = 0; i < num_nbr; i++) {
            int w = nbrs[i];
            if (levels[w] == INT_MAX) { // if not visited yet
                *finished = false;
                levels[w] = curr + 1;
            }
        }
    }
}


#endif
