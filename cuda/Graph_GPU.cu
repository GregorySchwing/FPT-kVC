#include "Graph_GPU.cuh"

__host__ Graph_GPU::Graph_GPU(const Graph & other): csr(other.csr),
    vertexCount(other.vertexCount){
//    hasntBeenRemoved.reserve(other.vertexCount);
//    verticesRemaining.reserve(other.vertexCount);
//    thrust_new_degrees_dev.reserve(other.vertexCount);
//    std::cout << "Copied" << std::endl;
}

