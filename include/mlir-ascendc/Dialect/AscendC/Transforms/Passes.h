//===- Passes.h - AscendC Patterns and Passes -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares passes on AscendC operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMS_PASSES_H
#define MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace func {
class FuncOp;
} // namespace func

namespace ascendc {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Creates a pass to test legal usage of AscendC features
std::unique_ptr<OperationPass<func::FuncOp>> createAscendCTestLegalityPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h.inc"

} // namespace ascendc
} // namespace mlir

#endif // MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMS_PASSES_H
