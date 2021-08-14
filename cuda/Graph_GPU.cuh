#ifndef GRAPH_GPU_H
#define GRAPH_GPU_H

#include "CSR_GPU.cuh"
#include "CSR.h"
#include "Graph.h"
#include <cuda.h>

class Graph;

template <typename T, size_t M, size_t N>
T (&make_sub_array(T (&orig)[M], size_t o))[N]
{
    return (T (&)[N])(*(orig + o));
}

class Graph_GPU {
public:
    /* Constructor to allocate induced subGraph_GPU G'' for each branch */
    __host__ __device__ Graph_GPU(const Graph & other);
    __host__ __device__ ~Graph_GPU();
    __host__ __device__ void InitTree(int treeSize, 
                                        int startingLevel, 
                                        int endingLevel, 
                                        Graph ** tree);
    __host__ __device__ long long CalculateSizeRequirement(int startingLevel,
                                                    int endingLevel);


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
