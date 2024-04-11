set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(ASCEND_LIBS
  MLIRAscendCDialect
)

set(LIBS
  ${dialect_libs}
  ${translation_libs}
  ${ASCEND_LIBS}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
)

add_mlir_tool(ascendc-mlir-translate
  ascendc-mlir-translate.cpp

  DEPENDS
  ${LIBS}
  )
llvm_update_compile_flags(ascendc-mlir-translate)
target_link_libraries(ascendc-mlir-translate PRIVATE ${LIBS})
mlir_check_link_libraries(ascendc-mlir-translate)
