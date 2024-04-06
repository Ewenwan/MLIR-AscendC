//===- translateAscendCToCpp.h - Helpers to create C++ emitter --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers to emit C++ code with the AscendC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_CPP_TRANSLATEASCENDCTOCPP_H
#define MLIR_TARGET_CPP_TRANSLATEASCENDCTOCPP_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {

/// Translates the given operation to C++ code. The operation or operations in
/// the region of 'op' need almost all be in AscendC dialect.
LogicalResult translateAscendCToCpp(Operation *op, raw_ostream &os,
                                    bool toggleAscendCSpecifics = false);

} // namespace mlir

#endif // MLIR_TARGET_CPP_TRANSLATEASCENDCTOCPP_H
