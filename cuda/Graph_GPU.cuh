#ifndef GRAPH_GPU_H
#define GRAPH_GPU_H

#include "CSR_GPU.cuh"
#include "CSR.h"
#include "Graph.h"
#include <cuda.h>

class Graph;

class Graph_GPU {
public:
    /* Constructor to allocate induced subGraph_GPU G'' for each branch */
    __host__ Graph_GPU(const Graph & other);
    __host__ __device__ ~Graph_GPU();


private:
    Graph_GPU * parent;
    int vertexCount;
    CSR_GPU csr;
    int * testVals;
    // vector of vectors of the children
    //thrust::device_vector< < int > childrenVertices;
    // The vertices passed as an argument to the InitGPrime method, used for creating an answer
    int * verticesToIncludeInCover;
    // set of vertices remaining, removed as vertices become degree 0
    int * verticesRemaining;
    // array of length vertexCount of booleans
    int * hasntBeenRemoved;
    // Set by SetEdgesLeftToCoverParallel method
    int edgesLeftToCover;
    
    // Following the CSR design pattern, a reference to the old degrees
    // For Original G, this comes from the ParallelKernel class
    int * old_degrees_ref;

    int * new_degrees;
    int * new_degrees_dev;
    
    friend class Graph;

};
#endif
