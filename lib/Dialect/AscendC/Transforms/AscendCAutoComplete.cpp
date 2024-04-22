//===----- AscendCAutoComplete.cpp - Code to insert enques and deques -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file inserts enques and deques that are AscendC specific based on the
// usage of localTensors.
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_ASCENDCAUTOCOMPLETE
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace ascendc;

namespace {

struct AscendCAutoCompletePass
    : public ascendc::impl::AscendCAutoCompleteBase<AscendCAutoCompletePass> {
  void runOnOperation() override;
};

void AscendCAutoCompletePass::runOnOperation() {
  // Collect alloc tensor ops
  SmallVector<AllocTensorOp> allocTensorOps;
  getOperation().walk([&](AllocTensorOp op) { allocTensorOps.push_back(op); });

  for (auto allocTensorOp : allocTensorOps) {
    OpBuilder ascendCBuilder(allocTensorOp->getContext());
    Location loc = allocTensorOp->getLoc();

    auto queue = allocTensorOp.getQueue();
    Value localTensor = allocTensorOp.getLocalTensor();

    for (auto *user : localTensor.getUsers()) {
      if (auto freeTensorOp = dyn_cast<FreeTensorOp>(user))
        continue;
      if (auto copyOp = dyn_cast<ascendc::DataCopyOp>(user)) {
        if (copyOp.getSrcTensor() == localTensor) {
          // UB->GM: insert enque after copy
          ascendCBuilder.setInsertionPoint(user);
          ascendCBuilder.create<DequeOp>(loc, localTensor.getType(), queue);
        } else {
          // GM->UB: insert deque before copy
          ascendCBuilder.setInsertionPointAfter(user);
          ascendCBuilder.create<EnqueOp>(loc, queue, localTensor);
        }
      } else {
        if (auto vaddOp = dyn_cast<ascendc::AddOp>(user)) {
          if (vaddOp.getDst() == localTensor) {
            // outputs -> insert enque after add
            ascendCBuilder.setInsertionPointAfter(user);
            ascendCBuilder.create<EnqueOp>(loc, queue, localTensor);
          } else {
            // inputs -> insert deque before add
            ascendCBuilder.setInsertionPoint(user);
            ascendCBuilder.create<DequeOp>(loc, localTensor.getType(), queue);
          }
          // TODO: replace add op operand with newly created from deque
        } else {
          llvm_unreachable("[ascendc-auto-complete]: unsupported area");
          return;
        }
      }
    }
  }
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::ascendc::createAscendCAutoCompletePass() {
  return std::make_unique<AscendCAutoCompletePass>();
}
