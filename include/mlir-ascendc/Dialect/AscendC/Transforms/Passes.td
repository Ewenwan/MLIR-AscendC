//===-- Passes.td - AscendC transformation definition file -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ASCENDC_TRANSFORMS_PASSES
#define MLIR_DIALECT_ASCENDC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AscendCTestLegality: Pass<"ascendc-test-legality", "func::FuncOp"> {
  let summary = "test legal usage of AscendC features";
  let description = [{
    This pass tests legal uses of AscendC features used in different operations
    such as whether the src and dst are legal in data copy op.
  }];
  let constructor = "mlir::ascendc::createAscendCTestLegalityPass()";
}

#endif // MLIR_DIALECT_ASCENDC_TRANSFORMS_PASSES
