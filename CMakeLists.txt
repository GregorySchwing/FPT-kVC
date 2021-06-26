cmake_minimum_required(VERSION 3.8)

project(FPT)

include_directories(serial)
include_directories(lib/boost/include)
include_directories(lib/boost)
include_directories(lib)


#Out-of-source build
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

IF(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

include_directories("${PROJECT_BINARY_DIR}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -stdlib=libc++") 
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -stdlib=libc++")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}") 
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_DEBUG /MTd /Zi /Ob0 /Od /RTC1")
    set(CMAKE_CXX_FLAGS_DEBUG_INIT "${CMAKE_CXX_FLAGS_DEBUG_INIT} /D_DEBUG /MTd /Zi /Ob0 /Od /RTC1")
    set(CMAKE_CXX_FLAGS_MINSIZEREL_INIT "${CMAKE_CXX_FLAGS_MINSIZEREL_INIT} /MT /O1 /Ob1 /D NDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_MINSIZEREL_INIT} /MT /O2 /Ob2 /D NDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE_INIT "${CMAKE_CXX_FLAGS_RELEASE_INIT} /MT /O2 /Ob2 /D NDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "${CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT} /MT /Zi /O2 /Ob1 /D NDEBUG")
endif()

# Set Source and Header files
include(${PROJECT_SOURCE_DIR}/CMake/FileLists.cmake)

# find Boost, and if not found download and install serialization library for checkpointing
# Set Source and Header files
#set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/FindBoost.cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
find_package(Boost)

if(Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: ${BOOST_INCLUDE_DIRS}")
endif()

include_directories(${BOOST_INCLUDE_DIRS})


# Find if CUDA exists and what is the version number
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    include(${PROJECT_SOURCE_DIR}/CMake/GPU.cmake)
endif()

# Setup Serial version
include(${PROJECT_SOURCE_DIR}/CMake/CPUSerial.cmake)
include(${PROJECT_SOURCE_DIR}/CMake/CPUParallel.cmake)

# find OpenMP and set it up
find_package(OpenMP)
if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
