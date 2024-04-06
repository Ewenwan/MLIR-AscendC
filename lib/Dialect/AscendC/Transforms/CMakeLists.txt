add_mlir_dialect_library(MLIRAscendCTransforms
  AscendCTestLegality.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AscendC

  DEPENDS
  MLIRAscendCPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRAscendC
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
)
