//===- InitAllDialects.h - MLIR Dialects Registration -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all
// ascendc-specific dialects to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_INITALLDIALECTS_H_
#define MLIRASCENDC_INITALLDIALECTS_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"

namespace mlir_ascendc {

/// Add all the ascendc-specific dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::ascendc::AscendCDialect>();
  // clang-format on
}

/// Append all the ascendc-specific dialects to the registry contained in the
/// given context.
inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  mlir_ascendc::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace mlir_ascendc

#endif // MLIRASCENDC_INITALLDIALECTS_H_
