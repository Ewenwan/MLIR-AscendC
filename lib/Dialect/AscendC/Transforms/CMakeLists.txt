add_mlir_dialect_library(MLIRAscendCTransforms
  AscendCAutoComplete.cpp
  AscendCBufferizationSimplification.cpp
  AscendCJustification.cpp
  ConvertToAscendC.cpp
  LegalizeFuncForAscendC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AscendC

  DEPENDS
  MLIRAscendCPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRAscendCDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRTransforms
)
