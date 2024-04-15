add_mlir_dialect_library(MLIRAscendCTransformOps
  AscendCTransformOps.cpp
  DialectExtension.cpp
  Promotion.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AscendC/TransformOps

  DEPENDS
  MLIRAscendCTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAscendCDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgTransformOps
  MLIRParser
  MLIRTransformDialect
  )
