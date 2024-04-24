get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(ascendc_libs
    MLIRAscendCDialect
    MLIRAscendCTransforms
    MLIRAscendCTransformOps
    MLIRAscendCToEmitC
    MLIRAscendCToCpp
)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${translation_libs}
  ${extension_libs}
  ${ascendc_libs}
)

add_mlir_tool(mlir-ascendc-compile
  mlir-ascendc-compile.cpp
  PassPipelines.cpp

  DEPENDS
  ${LIBS}
  )
llvm_update_compile_flags(mlir-ascendc-compile)
target_link_libraries(mlir-ascendc-compile PRIVATE ${LIBS})
target_link_libraries(mlir-ascendc-compile PRIVATE stdc++fs)
mlir_check_all_link_libraries(mlir-ascendc-compile)
