#include "Graph_GPU.cuh"


__host__ __device__ Graph_GPU::Graph_GPU(const Graph & other): csr(other.csr),
    vertexCount(other.vertexCount){

    //cudaMalloc(&hasntBeenRemoved, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&verticesRemaining, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&new_degrees_dev, other.vertexCount*sizeof(int)); 
}


__host__ __device__ void Graph_GPU::InitTree(long long treeSize, 
                                            long long edgesPerNode,
                                            long long numberOfVertices,
                                            int startingLevel, 
                                            int endingLevel, 
                                            Graph ** tree,
                                            int ** new_row_offsets_dev,
                                            int ** new_columns_dev,
                                            int ** values_dev,
                                            int ** new_degrees_dev){

    long long calculatedSizeReq = CalculateSizeRequirement(startingLevel, 
                                                            endingLevel);

    if (treeSize != calculatedSizeReq)
        printf("Asymmetric tree");
    else
        printf("Symmetric tree");



    

    
    //cudaMalloc(&hasntBeenRemoved, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&verticesRemaining, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&new_degrees_dev, other.vertexCount*sizeof(int)); 
}

__host__ __device__ Graph_GPU::~Graph_GPU(){
    //csr.~CSR_GPU();
    //cudaFree(hasntBeenRemoved);
    //cudaFree(verticesRemaining);
    //cudaFree(new_degrees_dev);
}

