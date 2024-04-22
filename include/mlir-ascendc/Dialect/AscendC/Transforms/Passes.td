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

def AscendCAutoComplete: Pass<"ascendc-auto-complete", "func::FuncOp"> {
  let summary = "Auto-complete AscendC instructions";
  let description = [{
    This pass automatically inserts enques and deques to a program targeted to
    AscendC.
  }];
  let constructor = "mlir::ascendc::createAscendCAutoCompletePass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
}

def LegalizeFuncForAscendC : Pass<"legalize-func-for-ascendc", "ModuleOp"> {
  let summary = "Legalize the functions within a module for AscendC Lowering";
  let description = [{
    This pass aims to take functions emitted from other Python frontends (such
    as Pytorch) and legalize their prototypes so that can be used for AscendC.
    This is ideally a temporary band-aid fix.

    This pass performs the following critical functions:

    1) It creates a new function that turns the original operands to GM_ADDR
    types while also keeps the original function. Therefore, this pass currently
    expects the user to run the inliner after running this pass.

    2) Converts the tensor operands and results into memref allocations with
    bufferization.to_tensor op.

    This pass is adapted from a pass created by Alex Singer in previous projects.
  }];

  let constructor = "mlir::ascendc::createLegalizeFuncForAscendCPass()";
  let dependentDialects = ["ascendc::AscendCDialect",
                           "func::FuncDialect",
                           "memref::MemRefDialect",
                           "bufferization::BufferizationDialect"];
}

def AscendCJustification: Pass<"ascendc-justification", "func::FuncOp"> {
  let summary = "Justify AscendC General Structures";
  let description = [{
    This pass validates the AscendC structures and make amendments to legalize
    AscendC usage such as adding TPipe and re-adjusting TPosition usage.
  }];
  let constructor = "mlir::ascendc::createAscendCJustificationPass();";
  let dependentDialects = ["ascendc::AscendCDialect",
                           "memref::MemRefDialect"];
}

def ConvertToAscendC: Pass<"convert-to-ascendc", "func::FuncOp"> {
  let summary = "Convert to AscendC Dialect";
  let description = [{
    This pass recursively converts MLIR code into AscendC specific instructions.
    Supports transformation on linalg.generic, memref.alloc, memref.copy.
    Future will add support to memref.subview.
  }];
  let constructor = "mlir::ascendc::createConvertToAscendCPass();";
  let dependentDialects = ["ascendc::AscendCDialect",
                           "memref::MemRefDialect",
                           "arith::ArithDialect"];
}

def AscendCBufferizationSimplification: Pass<"ascendc-bufferization-simplification", "func::FuncOp"> {
  let summary = "Simplify one specific bufferization pattern of memref usage";
  let description = [{
    This pass simpilifies the pattern generated after bufferization pass that a
    memref is created, copied in and immediately copied out. This pass is adapted
    from a pass created by Alex Singer in a previous project.

    This pass is intended to be applied after bufferization.
  }];
  let constructor = "mlir::ascendc::createAscendCBufferizationSimplificationPass();";
  let dependentDialects = ["ascendc::AscendCDialect",
                           "memref::MemRefDialect"];
}

#endif // MLIR_DIALECT_ASCENDC_TRANSFORMS_PASSES
