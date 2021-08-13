# Download and unpack googletest at configure time
configure_file(${PROJECT_SOURCE_DIR}/test/CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()
# Include file lists
include(${PROJECT_SOURCE_DIR}/test/FileList.cmake)
#MESSAGE(STATUS "Cuda version: ${PROJECT_SOURCE_DIR}/CMake/FileList.cmake")
#include(${PROJECT_SOURCE_DIR}/CMake/FileList.cmake)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(kVCTest ${headers_simple_parallel} ${sources_simple_parallel} ${headers_lib} ${sources_lib} ${TestHeaders} ${TestSources})
#add_executable(FPT_Test ${sources_simple_parallel} ${headers_simple_parallel} ${headers_lib} ${sources_lib} ${TestHeaders} ${TestSources})
target_link_libraries(kVCTest gtest_main gtest)
add_test(NAME kVCTest COMMAND kVCTest)
#set(GOMC_GTEST 1)
