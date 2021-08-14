#include "Graph_GPU.cuh"

__host__ __device__ Graph_GPU::Graph_GPU(const Graph & other): csr(other.csr),
    vertexCount(other.vertexCount){

    //cudaMalloc(&hasntBeenRemoved, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&verticesRemaining, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&new_degrees_dev, other.vertexCount*sizeof(int)); 
}

__host__ __device__ void Graph_GPU::InitTree(int treeSize, 
                                            int startingLevel, 
                                            int endingLevel, 
                                            Graph ** tree){
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

__host__ __device__ long long Graph_GPU::CalculateSizeRequirement(int startingLevel,
                                                        int endingLevel){
    long long summand= 0;
    // ceiling(vertexCount/2) loops
    for (int i = startingLevel; i < endingLevel; ++i)
        summand += pow (3.0, i);
    return summand;
}


__host__ __device__ Graph_GPU::~Graph_GPU(){
    //csr.~CSR_GPU();
    //cudaFree(hasntBeenRemoved);
    //cudaFree(verticesRemaining);
    //cudaFree(new_degrees_dev);
}

