//===- ascendc-mlir-opt.cpp - AscendC MLIR Optimizer Driver ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for ascendc-mlir-opt built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/InitAllDialects.h"
#include "mlir-ascendc/InitAllExtensions.h"
#include "mlir-ascendc/InitAllPasses.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir {
namespace test {
void registerTestTransformDialectInterpreterPass();
} // namespace test
} // namespace mlir

// This is used for AscendC lowering, MLIR_INCLUDE_TESTS should also be on.
void registerTestPasses() {
  mlir::test::registerTestTransformDialectInterpreterPass();
}

int main(int argc, char **argv) {
  // Register dialects.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir_ascendc::registerAllDialects(registry);

  // Register passes.
  mlir::registerAllPasses();
  mlir_ascendc::registerAllPasses();
  registerTestPasses();

  // Register extensions.
  mlir::registerAllExtensions(registry);
  mlir_ascendc::registerAllExtensions(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "AscendC MLIR optimizer driver\n", registry));
}
