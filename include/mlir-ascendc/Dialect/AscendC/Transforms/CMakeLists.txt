set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AscendC)
add_public_tablegen_target(MLIRAscendCPassIncGen)

add_mlir_doc(Passes AscendCPasses ./ -gen-pass-doc)
