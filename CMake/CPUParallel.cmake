
set(CPU_PARALLEL_name "K_VC_CPU_PARALLEL")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)
#add_executable(CPUParallel ${sources_cpu_parallel} ${headers_cpu_parallel} ${libHeaders} ${libSources} ${BOOST_INCLUDE_DIRS})
add_executable(CPUParallel ${headers_lib} ${sources_lib} ${sources_simple_parallel} ${headers_simple_parallel})
set_target_properties(CPUParallel PROPERTIES OUTPUT_NAME ${CPU_PARALLEL_name})
if(WIN32)
    #needed for hostname
    target_link_libraries(CPUParallel ws2_32)
endif()
