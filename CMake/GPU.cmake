# Find CUDA is enabled, set it up

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	message("-- Debug build type detected, passing : '-g -G --keep' to nvcc")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G --keep -lineinfo")
endif()


set(GEN_COMP_flag "-DFPT_CUDA -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT ")

if (GOMC_NVTX_ENABLED)
	message("-- Enabling profiling with NVTX for GPU")
	set(GEN_COMP_flag "${GEN_COMP_flag} -DGOMC_NVTX_ENABLED")
endif()


include_directories(cuda)
include_directories(lib)
include_directories(hybrid)


set(GPU_bin_flags "${GEN_COMP_flag}")

set(GPU_name "K_VC_GPU")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED true)

# Set host compiler
set(CCBIN "-ccbin=${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CCBIN}")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

message(STATUS "CUSPARSE HEADERS = ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUSPARSE LIB = ${CUDA_cusparse_LIBRARY}")

add_executable(GPU_bin ${headers_lib} ${sources_lib} ${main_cuda_parallel} ${headers_gpu} ${sources_gpu} ${sources_hybrid} ${headers_hybrid} ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(GPU_bin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME ${GPU_name}
    COMPILE_FLAGS "${GPU_bin_flags}")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("-- Debug build type detected, GPU_NVT setting CUDA_RESOLVE_DEVICE_SYMBOLS ON")
    set_property(TARGET GPU_bin PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
target_link_libraries( GPU_bin ${CUDA_cusparse_LIBRARY} ${CUDA_LIBRARIES})
if(WIN32)
    #needed for hostname
    target_link_libraries(GPU ws2_32)
endif()
