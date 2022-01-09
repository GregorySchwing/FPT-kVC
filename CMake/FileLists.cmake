
set(sources_cpu_serial 
    MainSerial.cpp
    serial/COO.cpp
    serial/CSR.cpp
    serial/DegreeController.cpp
    serial/Graph.cpp
    serial/NeighborsBinaryDataStructure.cpp
    serial/SequentialB1.cpp   
    serial/SequentialBuss.cpp
    serial/SequentialKernelization.cpp
    serial/SparseMatrix.cpp
   )

set(main_simple_parallel
    MainSimpleParallel.cpp
    )

set(main_cuda_parallel
    MainCUDAParallel.cpp
    )

set(sources_simple_parallel 
    simpleParallel/COO.cpp
    simpleParallel/CSR.cpp
    simpleParallel/Graph.cpp
    simpleParallel/SparseMatrix.cpp
    simpleParallel/ConnectednessTest.cpp
    simpleParallel/ParallelB1.cpp
    simpleParallel/ParallelKernelization.cpp
    simpleParallel/LinearTimeDegreeSort.cpp

   )

set(sources_gpu 
    cuda/ParallelB1_GPU.cu
   )

set(sources_hybrid 
    hybrid/COO.cpp
    hybrid/CSR.cpp
    hybrid/Graph.cpp
    hybrid/SparseMatrix.cpp
    hybrid/ConnectednessTest.cpp
    hybrid/ParallelB1.cpp
    hybrid/ParallelKernelization.cpp
    hybrid/LinearTimeDegreeSort.cpp
   )


set(headers_cpu_serial 
    serial/COO.h
    serial/CSR.h
    serial/DegreeController.h
    serial/Graph.h
    serial/NeighborsBinaryDataStructure.h
    serial/SequentialB1.h
    serial/SequentialBuss.h
    serial/SequentialKernelization.h
    serial/SparseMatrix.h
   )

set(headers_simple_parallel 
    simpleParallel/COO.h
    simpleParallel/CSR.h
    simpleParallel/Graph.h
    simpleParallel/SparseMatrix.h
    simpleParallel/ConnectednessTest.h
    simpleParallel/ParallelB1.h
    simpleParallel/ParallelKernelization.h
    simpleParallel/LinearTimeDegreeSort.h

   )

set(headers_gpu 
    cuda/CUDAUtils.cuh
    cuda/ParallelB1_GPU.cuh
   )

set(headers_hybrid
    hybrid/COO.h
    hybrid/CSR.h
    hybrid/Graph.h
    hybrid/SparseMatrix.h
    hybrid/ConnectednessTest.h
    hybrid/ParallelB1.h
    hybrid/ParallelKernelization.h
    hybrid/LinearTimeDegreeSort.h
   )


#set(libHeaders
#    lib/boost/include/dynamic_bitset.hpp
#    lib/boost/include/dynamic_bitset_fwd.hpp
#    lib/boost/include/dynamic_bitset/dynamic_bitset.hpp
#    lib/boost/include/dynamic_bitset/config.hpp
#   )

set(sources_lib
    lib/CSVIterator.cpp
    lib/CSVRange.cpp
    lib/CSVRow.cpp
    lib/DotWriter/lib/Attribute.cpp
    lib/DotWriter/lib/AttributeSet.cpp
    lib/DotWriter/lib/Cluster.cpp
    lib/DotWriter/lib/DotWriter.h
    lib/DotWriter/lib/Edge.cpp
    lib/DotWriter/lib/Enums.cpp
    lib/DotWriter/lib/Graph.cpp
    lib/DotWriter/lib/Idable.cpp
    lib/DotWriter/lib/IdManager.cpp
    lib/DotWriter/lib/Node.cpp
    lib/DotWriter/lib/RootGraph.cpp
    lib/DotWriter/lib/Subgraph.cpp
    lib/DotWriter/lib/Util.cpp
)

set(headers_lib
    lib/CSVIterator.h
    lib/CSVRange.h
    lib/CSVRow.h
    lib/DotWriter/lib/Attribute.h
    lib/DotWriter/lib/AttributeSet.h
    lib/DotWriter/lib/Cluster.h
    lib/DotWriter/lib/DotWriter.h
    lib/DotWriter/lib/Edge.h
    lib/DotWriter/lib/Enums.h
    lib/DotWriter/lib/Graph.h
    lib/DotWriter/lib/Idable.h
    lib/DotWriter/lib/IdManager.h
    lib/DotWriter/lib/Node.h
    lib/DotWriter/lib/RootGraph.h
    lib/DotWriter/lib/Subgraph.h
    lib/DotWriter/lib/Util.h
)

set(headers_lib_gpu
    lib/Random123/philox.h
)

set(cudaSources
    )

source_group("Header Files" FILES ${headers})
source_group("Lib Headers" FILES ${libHeaders})
source_group("CUDA Header Files" FILES ${cudaHeaders})
source_group("CUDA Source Files" FILES ${cudaSources})
