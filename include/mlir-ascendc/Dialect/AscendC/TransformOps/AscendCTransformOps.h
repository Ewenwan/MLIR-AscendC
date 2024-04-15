//===------ AscendCTransformOps.h - AscendC transform ops -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMOPS_ASCENDCTRANSFORMOPS_H
#define MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMOPS_ASCENDCTRANSFORMOPS_H

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

//===----------------------------------------------------------------------===//
// AscendC Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir-ascendc/Dialect/AscendC/TransformOps/AscendCTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace ascendc {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace ascendc
} // namespace mlir

#endif // MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMOPS_ASCENDCTRANSFORMOPS_H
