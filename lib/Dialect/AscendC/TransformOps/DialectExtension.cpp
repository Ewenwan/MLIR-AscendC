//===------ DialectExtension.cpp - Linalg transform dialect extension -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir-ascendc/Dialect/AscendC/TransformOps/AscendCTransformOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

using namespace mlir;

namespace {
class AscendCTransformDialectExtension
    : public transform::TransformDialectExtension<
          AscendCTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<linalg::LinalgDialect>();
    declareDependentDialect<ascendc::AscendCDialect>();

    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<memref::MemRefDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir-ascendc/Dialect/AscendC/TransformOps/AscendCTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::ascendc::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<AscendCTransformDialectExtension>();
}
