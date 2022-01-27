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

void CallPopulateTree(Graph & g, 
                     int root,
                     int * host_levels,
                    int * new_row_offs,
                    int * new_cols,
                    int * new_colors,
                    int * host_U){

    int * global_row_offsets_dev_ptr; // size N + 1
    int * global_degrees_dev_ptr; // size N, used for inducing the subgraph
    int * global_columns_dev_ptr; // size M
    int * global_values_dev_ptr; // on or off, size M
    int * global_levels; // size N, will contatin BFS level of nth node
    int * global_colors; // size N, will contatin color of nth node
    int * global_color_card;   // size N, will contatin size of color set
    // SSSP
    // Weights = size N
    int * global_W;
    // Mask = size M
    int * global_M;
    // Cost = size M
    int * global_C;
    // Update = size M
    int * global_U;

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

    // Vertex, Cols, Edge(on/off)
    cudaMalloc( (void**)&global_row_offsets_dev_ptr, (numberOfRows+1) * sizeof(int) );
    cudaMalloc( (void**)&global_columns_dev_ptr, numberOfEdgesPerGraph * sizeof(int) );
    cudaMalloc( (void**)&global_values_dev_ptr, numberOfEdgesPerGraph * sizeof(int) );
    cudaMalloc( (void**)&global_degrees_dev_ptr, (numberOfRows+1) * sizeof(int) );
    cudaMalloc( (void**)&global_levels, numberOfRows * sizeof(int) );
    cudaMalloc( (void**)&global_colors, numberOfRows * sizeof(int) );
    cudaMalloc( (void**)&global_color_card, numberOfRows * sizeof(int) );

    // SSSP
    cudaMalloc( (void**)&global_W, numberOfEdgesPerGraph * sizeof(int) );
    cudaMalloc( (void**)&global_M, numberOfRows * sizeof(int) );
    cudaMalloc( (void**)&global_C, numberOfRows * sizeof(int) );
    cudaMalloc( (void**)&global_U, numberOfRows * sizeof(int) );

    // Set all levels value to INT_MAX
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_levels),  INT_MAX, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_color_card),  1, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_W),  1, size_t(numberOfEdgesPerGraph));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_M),  0, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_C),  INT_MAX, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_U),  INT_MAX, size_t(numberOfRows));

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    int zero = 0;
    // Set root value to 0
    cudaMemcpy(&global_levels[root], &zero, 1 * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    CopyGraphToDevice(g,
                    numberOfEdgesPerGraph,
                    global_row_offsets_dev_ptr,
                    global_columns_dev_ptr,
                    global_values_dev_ptr,
                    global_degrees_dev_ptr,
                    global_colors);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
/*
    PerformBFS( numberOfRows,
                global_levels,
                global_row_offsets_dev_ptr,
                global_columns_dev_ptr);

    int k = 5;

    PerformBFSColoring( numberOfRows,
        k,
        global_levels,
        global_row_offsets_dev_ptr,
        global_columns_dev_ptr,
        global_colors,
        global_color_card);
        */
    PerformSSSP(numberOfRows,
                root,
                global_row_offsets_dev_ptr,
                global_columns_dev_ptr,
                global_W,
                global_M,
                global_C,
                global_U);
    
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    //cudaMemcpy(&host_levels[0], &global_levels[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&new_row_offs[0], &global_row_offsets_dev_ptr[0], (numberOfRows+1) * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&new_cols[0], &global_columns_dev_ptr[0], numberOfEdgesPerGraph * sizeof(int) , cudaMemcpyDeviceToHost);
    //cudaMemcpy(&new_colors[0], &global_colors[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_U[0], &global_U[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    cudaFree( global_row_offsets_dev_ptr );
    cudaFree( global_columns_dev_ptr );
    cudaFree( global_values_dev_ptr );
    cudaFree( global_degrees_dev_ptr );
    cudaFree( global_levels );
    cudaFree( global_colors );
    cudaFree( global_color_card );
    cudaFree( global_W );
    cudaFree( global_M );
    cudaFree( global_C );
    cudaFree( global_U );

    cudaDeviceSynchronize();
}

void CopyGraphToDevice( Graph & g,
                        int numberOfEdgesPerGraph,
                        int * global_row_offsets_dev_ptr,
                        int * global_columns_dev_ptr,
                        int * global_values_dev_ptr,
                        int * global_degrees_dev_ptr,
                        int * global_colors){

    int * new_degrees_ptr = thrust::raw_pointer_cast(g.GetNewDegRef().data());
    // allocate three device_vectors with 10 elements
    thrust::device_vector<int> colors(g.GetNumberOfRows());
    // initialize X to 0,1,2,3, ....
    thrust::sequence(colors.begin(), colors.end());
    int * new_colors_ptr = thrust::raw_pointer_cast(colors.data());

    std::cout << "remaining verts" << std::endl;
    for (auto & v : g.GetRemainingVerticesRef())
        std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "remaining verts size " << g.GetRemainingVerticesRef().size() << std::endl;
    
    // Degree CSR Data
    cudaMemcpy(global_degrees_dev_ptr, new_degrees_ptr, g.GetNumberOfRows() * sizeof(int),
                cudaMemcpyHostToDevice);
    cudaMemcpy(global_colors, new_colors_ptr, g.GetNumberOfRows() * sizeof(int),
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

void PerformBFSColoring(int numberOfRows,
                int k,
                int * global_levels,
                int * global_row_offsets_dev_ptr,
                int * global_columns_dev_ptr,
                int * global_colors,
                int * global_color_card){

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
            launch_gpu_bfs_coloring_kernel<<<oneThreadPerNode,threadsPerBlock>>>(
                                    numberOfRows,
                                    curr++, 
                                    k,
                                    global_levels,
                                    global_row_offsets_dev_ptr,
                                    global_columns_dev_ptr,
                                    global_colors,
                                    global_color_card,
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
                int * global_U){

        int oneThreadPerNode = (numberOfRows + threadsPerBlock - 1) / threadsPerBlock;
        int zero;
        int * finished = &zero;
        int * finished_gpu;

        cudaMalloc( (void**)&finished_gpu, 1 * sizeof(int) );
        cuMemsetD32(reinterpret_cast<CUdeviceptr>(&global_M[root]),  1, size_t(1));
        cuMemsetD32(reinterpret_cast<CUdeviceptr>(&global_C[root]),  0, size_t(1));
        cuMemsetD32(reinterpret_cast<CUdeviceptr>(&global_U[root]),  0, size_t(1));

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
                                    global_U);

            cudaDeviceSynchronize();
            checkLastErrorCUDA(__FILE__, __LINE__);

            launch_gpu_sssp_kernel_2<<<oneThreadPerNode,threadsPerBlock>>>(
                                    numberOfRows,
                                    global_row_offsets_dev_ptr,
                                    global_columns_dev_ptr,
                                    global_W,
                                    global_M,
                                    global_C,
                                    global_U);

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

__global__ void launch_gpu_bfs_kernel( int N, int curr, int *levels,
                                            int *nodes, int *edges, int * finished){
    int v = threadIdx.x;
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

__global__ void launch_gpu_bfs_coloring_kernel( int N, 
                                            int curr, 
                                            int k,
                                            int *levels,
                                            int *nodes, 
                                            int *edges, 
                                            int *colors,
                                            int * color_card,
                                            int * finished){
    int v = threadIdx.x;
    if (v >= N)
        return;
    int colored = 0;
    if (levels[v] == curr) {
        // iterate over neighbors
        int num_nbr = nodes[v+1] - nodes[v];
        int * nbrs = & edges[ nodes[v] ];
        for(int i = 0; i < num_nbr; i++) {
            int w = nbrs[i];
            if (levels[w] == INT_MAX) { // if not visited yet
                *finished = 0;
                levels[w] = curr + 1;
                if (!colored && color_card[colors[i]] < k){
                    colors[w] = colors[i];
                    color_card[colors[i]] = color_card[colors[i]] + 1;
                    colored = 1;
                }
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
                                            int * U){
    int v = threadIdx.x;
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
            }
        }
    }
}

__global__ void launch_gpu_sssp_kernel_2(   int N,       
                                            int * nodes,
                                            int * edges,
                                            int * W,
                                            int * M,
                                            int * C,
                                            int * U){
    int v = threadIdx.x;
    if (v >= N)
        return;
    else {
        if (C[v] > U[v]){
            C[v] = U[v];
            M[v] = true;
        }
        U[v] = C[v];
    }
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
