set(sources 
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

set(headers
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