set(sources_cpu_serial 
    Main.cpp
    serial/COO.cpp
    serial/CSR.cpp
    serial/DegreeController.cpp
    serial/Graph.cpp
    serial/NeighborsBinaryDataStructure.cpp
    serial/SequentialB1.cpp   
    serial/SequentialBuss.cpp
    serial/SequentialDFKernelization.cpp
    serial/SequentialKernelization.cpp
    serial/SparseMatrix.cpp
   )

set(sources_cpu_parallel 
    Main.cpp
    serial/COO.cpp
    serial/CSR.cpp
    serial/DegreeController.cpp
    serial/Graph.cpp
    serial/NeighborsBinaryDataStructure.cpp
    serial/SequentialB1.cpp   
    serial/SequentialBuss.cpp
    serial/SequentialDFKernelization.cpp
    serial/SequentialKernelization.cpp
    serial/SparseMatrix.cpp
   )

set(headers_cpu_serial 
    serial/COO.h
    serial/CSR.h
    serial/DegreeController.h
    serial/Graph.h
    serial/NeighborsBinaryDataStructure.h
    serial/SequentialB1.h
    serial/SequentialBuss.h
    serial/SequentialDFKernelization.h
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