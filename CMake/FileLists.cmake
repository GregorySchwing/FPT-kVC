
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
    Main.cu
    gpu/COO.cu
    gpu/CSR.cu
    gpu/Graph.cu
    gpu/SparseMatrix.cu
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
    simpleParallel/LinearTimeDegreeSort.cpp

   )

set(headers_gpu 
    gpu/COO.cuh
    gpu/CSR.cuh
    gpu/Graph.cuh
    gpu/SparseMatrix.cuh
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
)

set(headers_lib
    lib/CSVIterator.h
    lib/CSVRange.h
    lib/CSVRow.h
)

set(cudaHeaders
    )

set(cudaSources
    )

source_group("Header Files" FILES ${headers})
source_group("Lib Headers" FILES ${libHeaders})
source_group("CUDA Header Files" FILES ${cudaHeaders})
source_group("CUDA Source Files" FILES ${cudaSources})
