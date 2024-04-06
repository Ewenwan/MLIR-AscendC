add_mlir_dialect_library(MLIRAscendC
  AscendC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AscendC

  DEPENDS
  MLIRAscendCIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )
