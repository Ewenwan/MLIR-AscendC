# MLIR ASCENDC

This is the MLIR AscendC Project which contains relevant dialects, passes, tools and analysis.

This directory is set up similar to many MLIR projects. A good example of this is the standalone-dialect under mlir/examples or [SYCL MLIR](https://github.com/intel/llvm/tree/sycl-mlir/mlir-sycl)

## Build Instructions - Monolithic Build

Build MLIR AscendC as part of the LLVM Project with mlir enabled via the `LLVM_EXTERNAL_PROJECTS` mechanism.

```sh
mkdir build && cd build

cmake -G Ninja ../externals/llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=mlir-ascendc -DLLVM_EXTERNAL_MLIR_ASCENDC_SOURCE_DIR=.. \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_LIT_ARGS="-j32 -sv"
ninja check-mlir-ascendc
```
