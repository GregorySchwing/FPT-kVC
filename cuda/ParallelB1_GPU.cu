#ifdef FPT_CUDA

#include "ParallelB1_GPU.cuh"
#include <math.h>       /* pow */
// Sum of i = 0 to n/2
// 3^i



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


__host__ __device__ long long CalculateSizeRequirement(int startingLevel,
                                                        int endingLevel){
    long long summand= 0;
    // ceiling(vertexCount/2) loops
    for (int i = startingLevel; i < endingLevel; ++i)
        summand += pow (3.0, i);
    return summand;
}

__host__ __device__ long long CalculateLevelOffset(int level){

    // Closed form solution of partial geometric series
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    return (1.0 - pow(3.0, (level-1) + 1))/(1.0 - 3.0);

}

__device__ void AssignPointers(long long globalIndex,
                                long long edgesPerNode,
                                long long numberOfVertices,
                                Graph_GPU ** graphs,
                                int ** new_row_offsets_dev,
                                int ** new_columns_dev,
                                int ** values_dev,
                                int ** new_degrees_dev){

    (*graphs)[globalIndex].old_degrees_ref = (*graphs)[(globalIndex-1)/3].new_degrees_dev;
    (*graphs)[globalIndex].csr.old_column_indices_ref = (*graphs)[(globalIndex-1)/3].csr.new_column_indices_dev;
    (*graphs)[globalIndex].csr.old_row_offsets_ref = (*graphs)[(globalIndex-1)/3].csr.new_row_offsets_dev;
    (*graphs)[globalIndex].csr.old_values_ref = (*graphs)[(globalIndex-1)/3].csr.new_values_dev;

    (*graphs)[globalIndex].new_degrees_dev = new array_container(new_degrees_dev,
                                                          globalIndex,
                                                          numberOfVertices);
    (*graphs)[globalIndex].csr.new_row_offsets_dev = new array_container(new_row_offsets_dev,
                                                        globalIndex,
                                                        numberOfVertices+1);
    (*graphs)[globalIndex].csr.new_column_indices_dev = new array_container(new_columns_dev,
                                                    globalIndex,
                                                    edgesPerNode);
    (*graphs)[globalIndex].csr.new_values_dev = new array_container(values_dev,
                                                    globalIndex,
                                                    edgesPerNode);

}

__global__ void First_Graph_GPU(Graph_GPU * g_dev,
                                int vertexCount, 
                                int size,
                                int numberOfRows,
                                int ** old_row_offsets_dev,
                                int ** old_columns_dev,
                                int ** old_values_dev,
                                int ** new_row_offsets_dev,
                                int ** new_columns_dev,
                                int ** new_values_dev,
                                int ** old_degrees_dev,
                                int ** new_degrees_dev) {
     // notice this is how you use __device__ compiled code
     g_dev = new Graph_GPU(vertexCount, 
                    size,
                    numberOfRows,
                    old_row_offsets_dev,
                    old_columns_dev,
                    old_values_dev,
                    new_row_offsets_dev,
                    new_columns_dev,
                    new_values_dev,
                    old_degrees_dev,
                    new_degrees_dev);
     // use the sphere here


     return;
}

__global__ void CopyBackGraph(Graph_GPU * g_dev, int * internal_dev_ptr, int * sizedev2host){
    internal_dev_ptr = g_dev->new_degrees_dev->data;
    sizedev2host = &(g_dev->new_degrees_dev->count);
}


// Fill a perfect 3-ary tree to a given depth
__global__ void PopulateTreeParallelLevelWise_GPU(Graph_GPU * g,
                                                int numberOfLevels, 
                                                long long edgesPerNode,
                                                long long numberOfVertices,
                                                Graph_GPU ** graphs,
                                                int ** new_row_offsets_dev,
                                                int ** new_columns_dev,
                                                int ** values_dev,
                                                int ** new_degrees_dev){

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
        printf("Thread %lu, block %lu, vertexCount %d", leafIndex, myLevel, graphs[levelOffset + node]->vertexCount);
        AssignPointers(levelOffset + node,
                        edgesPerNode,
                        numberOfVertices,
                        graphs,
                        new_row_offsets_dev,
                        new_columns_dev,
                        values_dev,
                        new_degrees_dev
        );
    }
}


// Fill a perfect 3-ary tree to a given depth
__global__ void TearDownTree_GPU(int numberOfLevels, 
                                Graph_GPU ** graphs){

    printf("\nCalledTeardown");

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
        delete graphs[levelOffset + node];
    }
}

void CallPopulateTree(int numberOfLevels, 
                    Graph & g){


    //int treeSize = 200000;
    long long treeSize = CalculateSpaceForDesiredNumberOfLevels(numberOfLevels);
    long long expandedData = g.GetEdgesLeftToCover();
    long long condensedData = g.GetVertexCount();
    long long sizeOfSingleGraph = expandedData*2*sizeof(int) + 2*condensedData*sizeof(int);
    long long totalMem = sizeOfSingleGraph * treeSize;

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
        std::cout << '\n' << "Press a key to continue...; ctrl-c to terminate";
    } while (std::cin.get() != '\n');

    Graph_GPU * g_dev;

    cudaMalloc( (void**)&g_dev, 1 * sizeof(Graph_GPU) );

    CopyGraphToDevice(g, g_dev);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    Graph_GPU * graphs_ptr;
    int * new_row_offsets_dev_ptr;
    int * new_columns_dev_ptr;
    int * values_dev_ptr;
    int * new_degrees_dev_ptr; 
    
    cudaMalloc( (void**)&graphs_ptr, treeSize * sizeof(Graph_GPU) );
    cudaMalloc( (void**)&new_row_offsets_dev_ptr, ((g.GetVertexCount()+1)*treeSize) * sizeof(int) );
    cudaMalloc( (void**)&new_columns_dev_ptr, (g.GetEdgesLeftToCover()*treeSize) * sizeof(int) );
    cudaMalloc( (void**)&values_dev_ptr, (g.GetEdgesLeftToCover()*treeSize) * sizeof(int) );
    cudaMalloc( (void**)&new_degrees_dev_ptr, (g.GetVertexCount()*treeSize) * sizeof(int) );

    PopulateTreeParallelLevelWise_GPU<<<1,1>>>(g_dev,
                                        numberOfLevels, 
                                        g.GetEdgesLeftToCover(),
                                        g.GetVertexCount(),
                                        &graphs_ptr,
                                        &new_row_offsets_dev_ptr,
                                        &new_columns_dev_ptr,
                                        &values_dev_ptr,
                                        &new_degrees_dev_ptr);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    std::vector<int> mpt;
    /*
    InitGPrime_GPU<<<1,1,1>>>(g_dev, 
                            mpt, 
                            g.GetVerticesThisGraphIncludedInTheCover(), 
                            &graphs_ptr[0]);
    */
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
    
    TearDownTree_GPU<<<1,1,1>>>(numberOfLevels, &graphs_ptr);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);

    cudaFree( graphs_ptr );
    cudaFree( new_row_offsets_dev_ptr );
    cudaFree( new_columns_dev_ptr );
    cudaFree( values_dev_ptr );
    cudaFree( new_degrees_dev_ptr );

    cudaDeviceSynchronize();
}

__global__ void InitGPrime_GPU(Graph_GPU & g_dev, array_container * mpt, array_container * S, Graph_GPU * root){

}

void CopyGraphToDevice(Graph & g, Graph_GPU * g_dev){

    // Graph vectors
    //thrust::device_vector<int> old_degrees_dev(g.GetVertexCount());
   // thrust::device_vector<int> new_degrees_dev(g.GetVertexCount());
   // thrust::copy(g.GetOldDegRef().begin(), g.GetOldDegRef().end(), old_degrees_dev.begin());
   // thrust::copy(g.GetNewDegRef().begin(), g.GetNewDegRef().end(), new_degrees_dev.begin());
    thrust::device_vector<int> old_degrees_dev = g.GetOldDegRef();
    thrust::device_vector<int> new_degrees_dev = g.GetNewDegRef();
    //thrust::copy(g.GetOldDegRef().begin(), g.GetOldDegRef().end(), old_degrees_dev.begin());
    //thrust::copy(g.GetNewDegRef().begin(), g.GetNewDegRef().end(), new_degrees_dev.begin());
    //thrust::device_vector<int> old_degrees_dev = g.GetOldDegRef();



    // CSR vectors
    thrust::device_vector<int> new_row_offsets_dev = g.GetCSR().GetNewRowOffRef();
    thrust::device_vector<int> new_column_indices_dev = g.GetCSR().GetNewColRef();
    thrust::device_vector<int> old_row_offsets_dev = *(g.GetCSR().GetOldRowOffRef());
    thrust::device_vector<int> old_column_indices_dev = *(g.GetCSR().GetOldColRef());

    //thrust::copy(g.GetCSR().GetNewRowOffRef().begin(), g.GetCSR().GetNewRowOffRef().end(), new_row_offsets_dev.begin());
    //thrust::copy(g.GetCSR().GetNewColRef().begin(), g.GetCSR().GetNewColRef().end(), new_column_indices_dev.begin());
    /*  
    thrust::copy((g.GetCSR().GetOldRowOffRef())->begin(), (g.GetCSR().GetOldRowOffRef())->end(), old_row_offsets_dev.begin());
    thrust::copy((g.GetCSR().GetOldColRef())->begin(), (g.GetCSR().GetOldColRef())->end(), old_column_indices_dev.begin());
  
    // SparseMatrix vectors
    thrust::device_vector<int> new_values_dev;
    thrust::device_vector<int> old_values_dev;

    thrust::copy(g.GetCSR().GetNewValRef().begin(), g.GetCSR().GetNewValRef().end(), new_values_dev.begin());
    thrust::copy((g.GetCSR().GetOldValRef())->begin(), (g.GetCSR().GetOldValRef())->end(), old_values_dev.begin());
    

    // Graph pointers
    int * old_degrees_dev_ptr = thrust::raw_pointer_cast(old_degrees_dev.data());
    int * new_degrees_dev_ptr = thrust::raw_pointer_cast(new_degrees_dev.data());

    // CSR pointers
    int * new_row_offsets_dev_ptr = thrust::raw_pointer_cast(new_row_offsets_dev.data());
    int * new_column_indices_dev_ptr = thrust::raw_pointer_cast(new_column_indices_dev.data());
    int * old_row_offsets_dev_ptr = thrust::raw_pointer_cast(old_row_offsets_dev.data());
    int * old_column_indices_dev_ptr = thrust::raw_pointer_cast(old_column_indices_dev.data());
    
    // SparseMatrix pointers
    int * new_values_dev_ptr = thrust::raw_pointer_cast(new_values_dev.data());
    int * old_values_dev_ptr = thrust::raw_pointer_cast(old_values_dev.data());

    First_Graph_GPU<<<1,1>>>(g_dev,
                            1,
                            1,
                            1,
                            &old_row_offsets_dev_ptr,
                            &old_column_indices_dev_ptr,
                            &old_values_dev_ptr,
                            &new_row_offsets_dev_ptr,
                            &new_column_indices_dev_ptr,
                            &new_values_dev_ptr,
                            &old_degrees_dev_ptr,
                            &new_degrees_dev_ptr);
*/
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
    /*
    int * sizedev2host;
    int * new_values_dev2host_ptr;
    int * size;
    CopyBackGraph<<<1,1>>>(g_dev, new_values_dev2host_ptr, sizedev2host);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
    CubDebugExit(cudaMemcpy(&size, sizedev2host, 1*sizeof(int),
                          cudaMemcpyDeviceToHost));
    
    thrust::device_ptr<int> back2Host_ptr = thrust::device_pointer_cast(new_values_dev2host_ptr);
    thrust::device_vector<int> back2Host(back2Host_ptr, back2Host_ptr + (*size));
    */
    thrust::host_vector<int> hostFinal = new_degrees_dev;
    std::cout << "Priting data copied there and back" << std::endl;;

    //std::cout << "Size" << *size << std::endl;;
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
