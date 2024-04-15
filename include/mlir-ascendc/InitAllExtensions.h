//===- InitAllExtensions.h - MLIR Extension Registration --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialect
// extensions to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_INITALLEXTENSIONS_H_
#define MLIRASCENDC_INITALLEXTENSIONS_H_

#include "mlir-ascendc/Dialect/AscendC/TransformOps/AscendCTransformOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir_ascendc {

inline void registerAllExtensions(mlir::DialectRegistry &registry) {
  mlir::ascendc::registerTransformDialectExtension(registry);
}

} // namespace mlir_ascendc

#endif // MLIRASCENDC_INITALLEXTENSIONS_H_
