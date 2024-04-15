add_mlir_dialect_library(MLIRAscendCDialect
  AscendC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AscendC

  DEPENDS
  MLIRAscendCIncGen
  MLIRDeviceMappingInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )
