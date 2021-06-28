set(GPU_name "K_VC_GPU")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_executable(GPU_name ${sources_gpu} ${headers_gpu} ${libHeaders} ${libSources} ${BOOST_INCLUDE_DIRS})
set_target_properties(GPU PROPERTIES OUTPUT_NAME ${GPU_name})
if(WIN32)
    #needed for hostname
    target_link_libraries(GPU ws2_32)
endif()