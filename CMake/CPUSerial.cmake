#EnsemblePreprocessor defines NVT = 1, GEMC = 2, GCMC = 3, NPT = 4
#NPT (Isothermal-Isobaric) Ensemble

set(CPU_name "K_VC_CPU")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_executable(CPUSerial ${sources} ${headers} ${libHeaders} ${libSources} ${BOOST_INCLUDE_DIRS})
set_target_properties(CPUSerial PROPERTIES OUTPUT_NAME ${CPU_name})
if(WIN32)
    #needed for hostname
    target_link_libraries(CPUSerial ws2_32)
endif()
target_link_libraries( CPUSerial ${BOOST_LIBRARIES} )
