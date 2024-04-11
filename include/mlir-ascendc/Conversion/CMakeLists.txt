set(LLVM_TARGET_DEFINITIONS AscendCConversionPasses.td)
mlir_tablegen(AscendCConversionPasses.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(MLIRAscendCConversionPassIncGen)
