set(LLVM_TARGET_DEFINITIONS AscendCTransformOps.td)
mlir_tablegen(AscendCTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AscendCTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRAscendCTransformOpsIncGen)
