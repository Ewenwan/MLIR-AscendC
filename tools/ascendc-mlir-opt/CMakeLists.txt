set(ASCEND_LIBS
  MLIRAscendCDialect
  MLIRAscendCTransforms
  MLIRAscendCTransformOps
  MLIRAscendCToEmitC
)

set(LIBS
  MLIRMlirOptMain
  ${ASCEND_LIBS}
)

add_mlir_tool(ascendc-mlir-opt
  ascendc-mlir-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(ascendc-mlir-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(ascendc-mlir-opt)
