#ifndef GRAPH_GPU_H
#define GRAPH_GPU_H

#include "CSR_GPU.cuh"
#include "CSR.h"
#include "Graph.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class Graph;

class Graph_GPU {
public:
    /* Constructor to allocate induced subGraph_GPU G'' for each branch */
    __host__ Graph_GPU(const Graph & other);

private:
    Graph_GPU * parent;
    int vertexCount;
    CSR_GPU csr;
    thrust::device_vector<int> * testVals;
    // vector of vectors of the children
    //thrust::device_vector< < thrust::device_vector<int> > childrenVertices;
    // The vertices passed as an argument to the InitGPrime method, used for creating an answer
    thrust::device_vector<int> * verticesToIncludeInCover;
    // set of vertices remaining, removed as vertices become degree 0
    thrust::device_vector<int> * verticesRemaining;
    // array of length vertexCount of booleans
    thrust::device_vector<int> * hasntBeenRemoved;
    // Set by SetEdgesLeftToCoverParallel method
    int edgesLeftToCover;
    
    // Following the CSR design pattern, a reference to the old degrees
    // For Original G, this comes from the ParallelKernel class
    thrust::device_vector<int> * old_degrees_ref;

    thrust::host_vector<int> * thrust_new_degrees;
    thrust::device_vector<int> * thrust_new_degrees_dev;
    
    friend class Graph;

};
#endif
