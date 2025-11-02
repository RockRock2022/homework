cmake_minimum_required(VERSION 3.20)
project(pass)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(LLVM_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/llvm" CACHE PATH "Path to LLVMConfig.cmake")
set(MLIR_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/mlir" CACHE PATH "Path to MLIRConfig.cmake")


find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_definitions(-D__STDC_CONSTANT_MACROS)
add_definitions(-D__STDC_FORMAT_MACROS)
add_definitions(-D__STDC_LIMIT_MACROS)

add_subdirectory(src)
