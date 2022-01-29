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
                    int * host_U,
                    int * new_Pred){

    int * global_row_offsets_dev_ptr; // size N + 1
    int * global_degrees_dev_ptr; // size N, used for inducing the subgraph
    int * global_columns_dev_ptr; // size M
    int * global_values_dev_ptr; // on or off, size M
    int * global_levels; // size N, will contatin BFS level of nth node
    int * global_middle_vertex;
    int * global_colors; // size N, will contatin color of nth node
    int * global_color_card;   // size N, will contatin size of color set
    int * global_color_finished; // size N, contatin boolean of color finished, sometimes cardinality 1,2,3 is finished.
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
    // Malloc'ed by thrust
    //cudaMalloc( (void**)&global_colors, numberOfRows * sizeof(int) );
    // middle vertex flag can be removed if colors with
    // cardinality of size 3 that are non-cycles
    // are reset to original colors every time.
    // This might be worth experimenting with for a low-memory version.
    cudaMalloc( (void**)&global_middle_vertex, numberOfRows * sizeof(int) );
    cudaMalloc( (void**)&global_color_card, numberOfRows * sizeof(int) );
    cudaMalloc( (void**)&global_color_finished, numberOfRows * sizeof(int) );

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

    // Set all levels value to INT_MAX
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_levels),  INT_MAX, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_color_card),  1, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_middle_vertex),  0, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_color_finished),  0, size_t(numberOfRows));


    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_W),  1, size_t(numberOfEdgesPerGraph));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_M),  0, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_C),  INT_MAX, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_U),  INT_MAX, size_t(numberOfRows));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(global_Pred),  INT_MAX, size_t(numberOfRows));
    
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    int zero = 0;
    // Set root value to 0
    cudaMemcpy(&global_levels[root], &zero, 1 * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    // allocate three device_vectors with 10 elements
    thrust::device_vector<int> colors(g.GetNumberOfRows());
    // initialize X to 0,1,2,3, ....
    thrust::sequence(colors.begin(), colors.end());
    global_colors = thrust::raw_pointer_cast(colors.data());

    CopyGraphToDevice(g,
                    numberOfEdgesPerGraph,
                    global_row_offsets_dev_ptr,
                    global_columns_dev_ptr,
                    global_values_dev_ptr,
                    global_degrees_dev_ptr);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    // Root should be a degree 1 vertex.
    // We never color the root, so by starting the sssp from
    // degree 1 vertices, this isn't a problem.
    // Root's neighbot also must have color cardinality < k.
    // Finally, the next root should be chosen at a maximal 
    // depth from the previous root.
    // The SSSP and Color algorithm ends when either the entire graph is colored
    // or no such vertices remain.
    PerformSSSP(numberOfRows,
                root,
                global_row_offsets_dev_ptr,
                global_columns_dev_ptr,
                global_W,
                global_M,
                global_C,
                global_U,
                global_Pred,
                global_U_Pred);
    
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    int k = 4;
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
                            global_row_offsets_dev_ptr,
                            global_columns_dev_ptr,
                            global_middle_vertex,
                            global_M,
                            global_U,
                            global_U_Pred,
                            global_colors,
                            global_color_card,
                            global_color_finished);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);


    //cudaMemcpy(&host_levels[0], &global_levels[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&new_row_offs[0], &global_row_offsets_dev_ptr[0], (numberOfRows+1) * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&new_cols[0], &global_columns_dev_ptr[0], numberOfEdgesPerGraph * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&new_colors[0], &global_colors[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_U[0], &global_U[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy(&new_Pred[0], &global_U_Pred[0], numberOfRows * sizeof(int) , cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    cudaFree( global_row_offsets_dev_ptr );
    cudaFree( global_columns_dev_ptr );
    cudaFree( global_values_dev_ptr );
    cudaFree( global_degrees_dev_ptr );
    cudaFree( global_levels );
    //cudaFree( global_colors );
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
                            int * global_U,
                            int * global_U_Pred,
                            int * global_colors,
                            int * global_color_card,
                            int * global_color_finished){
    int oneThreadPerNode = (numberOfRows + threadsPerBlock - 1) / threadsPerBlock;
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
    if (v >= N || color_finished[v])
        return;
    int cv = colors[v];
    int myOwnColorEdges = 0;
    int middleVertex;
    // Guaruntees we need another vertex and v is not an internal vertex
    // for example, with path a - b - c; we guaruntee v is not b.
    if (color_card[v] == 3) {
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
    if (v >= N || color_finished[v])
        return;
    int cv = colors[v];
    int middleVertexCount = 0;
    int foundCycle;
    // Guaruntees we need another vertex and v is not an internal vertex
    // for example, with path a - b - c; we guaruntee v is not b.
    if (color_card[cv] == 3) {
        // iterate over neighbors
        int num_nbr = nodes[v+1] - nodes[v];
        int * nbrs = & edges[ nodes[v] ];
        for(int i = 0; i < num_nbr; i++) {
            int w = nbrs[i];
            int mvw = middle_vertex[w];
            if (mvw){
                ++middleVertexCount;
            }
        }
        foundCycle = middleVertexCount > 1;
        color_finished[cv] = foundCycle;
    } else if (color_card[cv] == 4){
        color_finished[cv] = true;
    }
}

__global__ void launch_gpu_combine_colors_kernel( int N,
                                                int k,
                                                int iter,
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
    int cv = colors[v];
    // Guaruntees we need another vertex and v is not an internal vertex
    // for example, with path a - b - c; we guaruntee v is not b.
    if (color_card[v] < k && color_card[v] % ((U[v] + iter) % k) == 0) {
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
    if (v >= N || color_finished[v] || middle_vertex[v])
        return;
    if ((U[v] + iter) % k != 0 || U[v] == INT_MAX)
        return;
    int w = U_Pred[v];
    // Race condition, but the kernel ends, so we get synchronization
    // it doesn't matter who wins, we need to initiate change in the graph.
    if (!color_finished[w] && !middle_vertex[w]){
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
    if (v >= N || color_finished[v] || middle_vertex[v])
        return;
    if ((U[v] + iter) % k != 0 || U[v] == INT_MAX)
        return;

    int w = U_Pred[v];
    int wc;
    if (M[w]){
        wc = colors[w];
        color_card[wc] = color_card[wc] + 1;
        M[w] = 0;
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

// Not currently working since we reset the marks to 0.
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
    if (v >= N || color_finished[v] || middle_vertex[v])
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
    if (vcc > wcc && !middle_vertex[w]){
        colors[w] = colors[v];
        M[w] = 1;
    }
}
#endif
