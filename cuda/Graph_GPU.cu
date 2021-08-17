#ifdef FPT_CUDA


#include "Graph_GPU.cuh"



__host__ __device__ Graph_GPU::Graph_GPU(int vertexCount): csr(vertexCount),
    vertexCount(vertexCount){

    //cudaMalloc(&hasntBeenRemoved, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&verticesRemaining, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&new_degrees_dev, other.vertexCount*sizeof(int)); 
}

__host__ __device__ Graph_GPU::Graph_GPU(const Graph & other): csr(other.csr),
    vertexCount(other.vertexCount){

    //cudaMalloc(&hasntBeenRemoved, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&verticesRemaining, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&new_degrees_dev, other.vertexCount*sizeof(int)); 
}

__host__ __device__ Graph_GPU::Graph_GPU(int vertexCount, 
                                        int size,
                                        int numberOfRows,
                                        int * old_row_offsets_dev,
                                        int * old_columns_dev,
                                        int * old_values_dev,
                                        int * new_row_offsets_dev,
                                        int * new_columns_dev,
                                        int * new_values_dev,
                                        int * old_degrees_dev,
                                        int * new_degrees_dev): 
    csr(vertexCount, size, numberOfRows, old_row_offsets_dev,
        old_columns_dev, old_values_dev, new_row_offsets_dev,
        new_columns_dev, new_values_dev),
    vertexCount(vertexCount){

    this->old_degrees_ref = new array_container(old_values_dev, 0, vertexCount);
    this->new_degrees_dev = new array_container(new_degrees_dev, 0, vertexCount);


    //cudaMalloc(&hasntBeenRemoved, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&verticesRemaining, other.vertexCount*sizeof(int)); 
    //cudaMalloc(&new_degrees_dev, other.vertexCount*sizeof(int)); 
}

__host__ __device__ void Graph_GPU::InitTree(Graph & root,
                                            long long treeSize, 
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


}

__host__ __device__ Graph_GPU::~Graph_GPU(){
    delete new_degrees_dev;
    delete csr.new_row_offsets_dev;
    delete csr.new_column_indices_dev;
    delete csr.new_values_dev;
}

#endif
