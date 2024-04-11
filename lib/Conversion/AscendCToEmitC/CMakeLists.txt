add_mlir_conversion_library(MLIRAscendCToEmitC
  AscendCToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AscendCToEmitC

  DEPENDS
  MLIRAscendCConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIREmitCDialect
  MLIRTransforms
  MLIRAscendCDialect
  )
