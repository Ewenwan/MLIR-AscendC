//===- AscendCTestLegality.cpp - Code to test dialect set up structure ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file test legal uses of AscendC specific features, currently supports
// legality check for DataCopy TPosition.
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_ASCENDCTESTLEGALITY
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

#define DEBUG_TYPE "ascendc-test-legality"

using namespace mlir;
using namespace ascendc;

namespace {

struct AscendCTestLegalityPass
    : public ascendc::impl::AscendCTestLegalityBase<AscendCTestLegalityPass> {
  void runOnOperation() override;
};

static TPosition getTensorTposition(Value tensor) {
  if (dyn_cast<GlobalTensorType>(tensor.getType()))
    return TPosition::GM;
  if (auto allocTensor = dyn_cast<AllocTensorOp>(tensor.getDefiningOp()))
    return allocTensor.getQueue().getType().getTposition();
  return TPosition::MAX;
}

void AscendCTestLegalityPass::runOnOperation() {
  // Collect all data copy ops
  SmallVector<DataCopyOp> dataCopyOps;
  getOperation().walk([&](DataCopyOp op) { dataCopyOps.push_back(op); });

  for (auto dataCopy : dataCopyOps) {
    TPosition dstPosition = getTensorTposition(dataCopy.getDstTensor());
    TPosition srcPosition = getTensorTposition(dataCopy.getSrcTensor());

    // Unfound TPosition
    if (dstPosition == TPosition::MAX || srcPosition == TPosition::MAX) {
      mlir::emitError(dataCopy->getLoc()) << "Unfound TPosition";
      signalPassFailure();
    }

    // Supported: GM->A1, GM->B1, GM->VECIN
    if (srcPosition == TPosition::GM) {
      if (dstPosition == TPosition::A1 || dstPosition == TPosition::B1 ||
          dstPosition == TPosition::VECIN) {
        continue;
      }
    }
    // Supported: CO2->GM, VECOUT->GM
    if (dstPosition == TPosition::GM) {
      if (srcPosition == TPosition::CO2 || srcPosition == TPosition::VECOUT) {
        continue;
      }
    }
    // Supported: CO1->CO2, VECIN->VECOUT
    if (srcPosition == TPosition::CO1 && dstPosition == TPosition::CO2)
      continue;
    if (srcPosition == TPosition::VECIN && dstPosition == TPosition::VECOUT)
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Unsupported data copy path from " << srcPosition
                            << " to " << dstPosition << "\n");
    mlir::emitError(dataCopy->getLoc()) << "unsupported data copy path";
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::ascendc::createAscendCTestLegalityPass() {
  return std::make_unique<AscendCTestLegalityPass>();
}
