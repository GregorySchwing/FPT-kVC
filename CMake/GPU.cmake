set(GPU_name "K_VC_GPU")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)
message(STATUS "CUSPARSE HEADERS = ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUSPARSE LIB = ${CUDA_cusparse_LIBRARY}")
add_executable(GPU_bin ${sources_gpu} ${headers_gpu} ${CUDAToolkit_INCLUDE_DIRS} ${libHeaders} ${libSources} ${BOOST_INCLUDE_DIRS})
set_target_properties(GPU_bin PROPERTIES OUTPUT_NAME ${GPU_name})
target_link_libraries( GPU_bin ${CUDA_cusparse_LIBRARY} )
if(WIN32)
    #needed for hostname
    target_link_libraries(GPU ws2_32)
endif()