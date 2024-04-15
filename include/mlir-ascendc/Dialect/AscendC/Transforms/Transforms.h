//===- Transforms.h - AscendC Patterns and Transformations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares AscendC transformations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMS_TRANSFORMS_H
#define MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Transforms
//===----------------------------------------------------------------------===//

// This a function used for AscendC promotion originally created by Xingyu Yue
// for previous projects. This legalizes the operands for promotion operation.
LogicalResult
promoteSubviewsAscendCPrecondition(Operation *op, OpBuilder &b,
                                   linalg::LinalgPromotionOptions options);

/// Allocate the subview in the AscendC UB memory.
std::optional<Value> allocateAscendCVECIN(OpBuilder &builder,
                                          memref::SubViewOp subview,
                                          ArrayRef<Value> sizeBounds,
                                          DataLayout &);

/// Normal copy to between src and dst.
LogicalResult copyToAscendCVECIN(OpBuilder &b, Value src, Value dst);

/// Deallocate AscendC UB memory
LogicalResult deallocateAscendCVECIN(OpBuilder &, Value /*buffer*/);

} // namespace ascendc
} // namespace mlir

#endif // MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMS_TRANSFORMS_H
