//===--- AscendCJustification.cpp - Code to validate AscendC structure ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass validates the AscendC structures and make amendments to legalize
// AscendC usage such as adding TPipe and re-adjust TPosition usage.
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include <cassert>

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_ASCENDCJUSTIFICATION
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

#define DEBUG_TYPE "ascendc-justification"

using namespace mlir;
using namespace ascendc;

namespace {

struct AscendCJustificationPass
    : public ascendc::impl::AscendCJustificationBase<AscendCJustificationPass> {
  void runOnOperation() override;
};

void AscendCJustificationPass::runOnOperation() {
  func::FuncOp func = getOperation();
  Location loc = func->getLoc();

  // Create TPipeOp at the beginning of the function.
  OpBuilder builder(func.getRegion());
  builder.create<CreatePipeOp>(loc, TPipeType::get(&getContext()));

  // Adjust AscendC TPosition of the generic outputs. Currently only supports
  // vector op so is fixed to VECOUT.
  SmallVector<linalg::GenericOp> genericOps;
  func->walk([&](linalg::GenericOp op) { genericOps.push_back(op); });
  assert(genericOps.size() == 1 &&
         "[ascendc-justification]: only supports one generic");
  for (auto genericOp : genericOps) {
    assert(genericOp.getOutputs().size() == 1 &&
           "expecting only one localTensor as output");
    Value output = genericOp.getOutputs().front();
    MemRefType outputMemRef = dyn_cast<MemRefType>(output.getType());
    assert(outputMemRef && "output should be a memref");
    memref::AllocOp allocOp = dyn_cast<memref::AllocOp>(output.getDefiningOp());
    assert(allocOp && "output memref should be defined by a alloc op");

    OpBuilder builder(allocOp);
    Location loc = allocOp->getLoc();
    MemRefType newMemref =
        MemRefType::get(outputMemRef.getShape(), outputMemRef.getElementType(),
                        outputMemRef.getLayout(),
                        TPositionAttr::get(&getContext(), TPosition::VECOUT));
    Value newAlloc = builder.create<memref::AllocOp>(loc, newMemref);
    output.replaceAllUsesWith(newAlloc);
    allocOp->erase();
  }
}

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createAscendCJustificationPass() {
  return std::make_unique<AscendCJustificationPass>();
}
