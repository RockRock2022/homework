set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

set(LLVM_TARGET_DEFINITIONS adjust_layout.td)
mlir_tablegen(adjust_layout.hpp.inc -gen-pass-decls)

add_custom_target(adjust-layout-inc DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/adjust_layout.hpp.inc
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(SOURCES
  adjust_layout.cpp
)

add_library(MyMLIRTosaTransforms SHARED ${SOURCES})

target_include_directories(MyMLIRTosaTransforms
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(MyMLIRTosaTransforms
  PRIVATE
  MLIRTosaDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTransforms
  MLIRRewrite
  MLIRPass
  MLIRIR
  MLIRSupport
)

add_dependencies(MyMLIRTosaTransforms adjust-layout-inc)

add_executable(my-opt my-opt.cpp)
set_target_properties(my-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

target_include_directories(my-opt
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add all required MLIR libraries
target_link_libraries(my-opt
  PRIVATE
  MLIROptLib
  MLIRPass
  MLIRIR
  MLIRSupport
  MLIRTosaDialect
  MLIRTosaTransforms  
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRRewrite
  MyMLIRTosaTransforms
)

add_dependencies(my-opt adjust-layout-inc)
