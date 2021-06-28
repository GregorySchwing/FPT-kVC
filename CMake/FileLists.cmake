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

set(sources_cpu_parallel 
    MainOpenMP.cpp
    openmp/COO.cpp
    openmp/CSR.cpp
    openmp/DegreeController.cpp
    openmp/Graph.cpp
    openmp/NeighborsBinaryDataStructure.cpp
    openmp/ParallelB1.cpp   
    openmp/ParallelBuss.cpp
    openmp/ParallelKernelization.cpp
    openmp/SparseMatrix.cpp
   )

set(sources_gpu 
    Main.cu
    gpu/COO.cu
    gpu/CSR.cu
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

set(headers_cpu_parallel 
    openmp/COO.h
    openmp/CSR.h
    openmp/DegreeController.h
    openmp/Graph.h
    openmp/NeighborsBinaryDataStructure.h
    openmp/ParallelB1.h
    openmp/ParallelBuss.h
    openmp/ParallelKernelization.h
    openmp/SparseMatrix.h
   )

set(headers_gpu 
    gpu/COO.cuh
    gpu/CSR.cuh
   )

set(libHeaders
    lib/boost/include/dynamic_bitset.hpp
    lib/boost/include/dynamic_bitset_fwd.hpp
    lib/boost/include/dynamic_bitset/dynamic_bitset.hpp
    lib/boost/include/dynamic_bitset/config.hpp
   )

set(libSources
    )

set(cudaHeaders
    )

set(cudaSources
    )

source_group("Header Files" FILES ${headers})
source_group("Lib Headers" FILES ${libHeaders})
source_group("CUDA Header Files" FILES ${cudaHeaders})
source_group("CUDA Source Files" FILES ${cudaSources})