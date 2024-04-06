# MLIR AscendC Project
project(mlir-ascendc LANGUAGES CXX C)

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_INCLUDE_DIRS ${MLIR_MAIN_SRC_DIR}/include)
set(MLIR_CMAKE_DIR ${MLIR_MAIN_SRC_DIR}/cmake/modules)
set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
set(ASCENDC_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir-ascendc/include)

include_directories(${MLIR_TABLEGEN_OUTPUT_DIR})
include_directories(${ASCENDC_TABLEGEN_OUTPUT_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MLIR_ASCENDC_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})

add_definitions(${LLVM_DEFINITIONS})

set(MLIR_ASCENDC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_ASCENDC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MLIR_ASCENDC_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(MLIR_ASCENDC_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(include/mlir-ascendc)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

install(DIRECTORY include/
  DESTINATION include
  COMPONENT mlir-ascendc-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "*.sv"
  PATTERN "LICENSE.TXT"
  )

install(DIRECTORY ${PROJECT_BINARY_DIR}/include/
  DESTINATION include
  COMPONENT mlir-ascendc-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
  )
