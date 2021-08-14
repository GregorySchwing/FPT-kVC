#include "Graph_GPU.cuh"

Graph_GPU::Graph_GPU(const Graph_GPU & other): csr(other.csr),
    vertexCount(other.vertexCount){
    hasntBeenRemoved.reserve(other.vertexCount);
    verticesRemaining.reserve(other.vertexCount);
    new_degrees.reserve(other.vertexCount);
//    std::cout << "Copied" << std::endl;
}