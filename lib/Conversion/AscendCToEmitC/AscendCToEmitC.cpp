//===- AscendCToEmitC.cpp - AscendC to EmitC dialect conversion ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert the AscendC dialect to the EmitC
// Dialect for cpp translation.
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Conversion/AscendCToEmitC/AscendCToEmitC.h"
#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#define DEBUG_TYPE "ascendc-to-emitc"
#define DBGLINE() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: " << (X) << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTASCENDCTOEMITC
#include "mlir-ascendc/Conversion/AscendCConversionPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

class ConvertAscendCAllocTensorToEmitC
    : public OpRewritePattern<ascendc::AllocTensorOp> {
public:
  using OpRewritePattern<ascendc::AllocTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ascendc::AllocTensorOp op,
                                PatternRewriter &rewriter) const final {
    LDBG("HiHi");
    op->dump();
    return success();
  }
};

void mlir::ascendc::populateAscendCToEmitCConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertAscendCAllocTensorToEmitC>
               (patterns.getContext());
  // clang-format on
}

namespace {
struct ConvertAscendCToEmitCPass
    : public impl::ConvertAscendCToEmitCBase<ConvertAscendCToEmitCPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertAscendCToEmitCPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, arith::ArithDialect,
                         scf::SCFDialect>();
  // clang-format off
  target.addIllegalOp<ascendc::AllocTensorOp>();
  // clang-format on
  RewritePatternSet patterns(&getContext());
  populateAscendCToEmitCConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertAscendCToEmitCPass() {
  return std::make_unique<ConvertAscendCToEmitCPass>();
}
