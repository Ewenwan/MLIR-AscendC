//===-- ConvertToAscendC.cpp - Code to convert prgram to AscendC space ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass recursively converts operations into AscendC space. This is made as
// a transformation pass due to specialized types required by AscendC Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_CONVERTTOASCENDC
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

#define DEBUG_TYPE "convert-to-ascendc"
#define DBGLINE() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: " << (X) << "\n")

using namespace mlir;
using namespace ascendc;

namespace {

// Creates a constant I32 type.
static Value constantI32(OpBuilder &builder, Location loc, int64_t i) {
  return builder.create<arith::ConstantIntOp>(loc, i, 32);
}

// Convert AllocOp to ascendc AllocTensorOp
static AllocTensorOp convertAllocOp(memref::AllocOp allocOp, Value pipe) {
  OpBuilder builder(allocOp);
  Location loc = allocOp->getLoc();

  MemRefType allocRes = allocOp.getResult().getType();
  assert(allocRes.getRank() == 1 && "AscendC Operates on 1 dimension.");

  if (isa<TPositionAttr>(allocRes.getMemorySpace())) {
    // restrict buf num to 1 now.
    int64_t bufNum = 1;
    auto tposition = cast<TPositionAttr>(allocRes.getMemorySpace()).getValue();

    // Create tile_length from alloc memref shape
    Value tileLength = constantI32(builder, loc, allocRes.getShape().front());
    Value queue = builder.create<CreateQueueOp>(
        loc, TQueType::get(allocOp.getContext(), tposition, bufNum), pipe,
        tileLength);
    AllocTensorOp localTensorOp = builder.create<ascendc::AllocTensorOp>(
        loc,
        LocalTensorType::get(allocRes.getShape(), allocRes.getElementType()),
        queue);
    return localTensorOp;
  }
  return nullptr;
}

static void convertCopyOp(memref::CopyOp copyOp, bool gm2ub, Operation *tpipeOp,
                          AllocTensorOp localTensorOp,
                          MutableArrayRef<BlockArgument> &gmAddr) {
  BaseMemRefType gmMemRef =
      gm2ub ? copyOp.getSource().getType() : copyOp.getTarget().getType();
  assert(gmMemRef.getMemorySpace() && "all memref should have memspace now");
  if (isa<TPositionAttr>(gmMemRef.getMemorySpace())) {
    TPosition tposition =
        cast<TPositionAttr>(gmMemRef.getMemorySpace()).getValue();
    assert(tposition == TPosition::GM && "Copy source should be GM");
    OpBuilder globalTensorBuilder(tpipeOp);
    Value tileLength = globalTensorBuilder.create<arith::ConstantIntOp>(
        tpipeOp->getLoc(), gmMemRef.getShape().front(), 32);
    Value globalTensor = globalTensorBuilder.create<CreateGlobalTensorOp>(
        tpipeOp->getLoc(),
        GlobalTensorType::get(gmMemRef.getShape(), gmMemRef.getElementType()),
        gmAddr.front(), Value(), tileLength);
    gmAddr = gmAddr.drop_front();

    OpBuilder copyBuilder(copyOp);
    if (gm2ub)
      copyBuilder.create<DataCopyOp>(copyOp->getLoc(),
                                     localTensorOp.getLocalTensor(), Value(),
                                     globalTensor, Value(), tileLength);
    else
      copyBuilder.create<DataCopyOp>(copyOp->getLoc(), globalTensor, Value(),
                                     localTensorOp.getLocalTensor(), Value(),
                                     tileLength);
    copyOp->erase();
  } else {
    llvm_unreachable("Expecting TPosition as memspace");
    return;
  }
}

static void convertDeallocOp(memref::DeallocOp deallocOp,
                             AllocTensorOp localTensorOp) {
  OpBuilder freeTensorbuilder(deallocOp);
  freeTensorbuilder.create<FreeTensorOp>(deallocOp->getLoc(),
                                         localTensorOp.getQueue(),
                                         localTensorOp.getLocalTensor());
  deallocOp->erase();
}

static void convertGenericOp(linalg::GenericOp genericOp,
                             SmallVector<Value> genericNewInputs,
                             SmallVector<Value> genericNewOutputs) {
  Location loc = genericOp->getLoc();
  OpBuilder genericBuilder(genericOp);
  SmallVector<Operation *> linalgRegionOps;
  genericOp.getRegion().walk(
      [&](Operation *op) { linalgRegionOps.push_back(op); });

  for (auto *op : linalgRegionOps) {
    if (dyn_cast<linalg::YieldOp>(op))
      continue;
    if (dyn_cast<arith::AddFOp>(op) || dyn_cast<arith::AddIOp>(op)) {
      auto allocTensorOp =
          dyn_cast<AllocTensorOp>(genericNewInputs.front().getDefiningOp());
      assert(allocTensorOp && "expect localTensor as generic inputs and "
                              "allocted by allocTensorOp");
      auto createQueue =
          dyn_cast<CreateQueueOp>(allocTensorOp.getQueue().getDefiningOp());
      assert(createQueue && "expect a create queue from localtensor queue");
      Value tileLength = createQueue.getTileLength();
      genericBuilder.create<ascendc::AddOp>(loc, genericNewInputs[0],
                                            genericNewInputs[1], tileLength,
                                            genericNewOutputs[0]);
    } else {
      llvm_unreachable("unreconginzied op in linalg region");
      return;
    }
  }
  genericOp->erase();
}

struct ConvertToAscendCPass
    : public ascendc::impl::ConvertToAscendCBase<ConvertToAscendCPass> {
  void runOnOperation() override;
};

void ConvertToAscendCPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Validate if TPipe exists.
  CreatePipeOp tpipeOp = nullptr;
  func->walk([&](Operation *op) {
    if (auto pipeOp = dyn_cast<CreatePipeOp>(op)) {
      tpipeOp = pipeOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  assert(tpipeOp && "TPipe is required for AscendC Program.");
  MutableArrayRef<BlockArgument> gmAddr = func.getArguments();

  // Collect and start with generic ops.
  SmallVector<linalg::GenericOp> genericOps;
  SmallVector<Value> genericNewInputs;
  SmallVector<Value> genericNewOutputs;
  func->walk([&](linalg::GenericOp op) { genericOps.push_back(op); });

  if (genericOps.empty())
    return;
  for (auto genericOp : genericOps) {
    SmallVector<Value> args = genericOp.getInputs();
    args.append(SmallVector<Value>(genericOp.getOutputs()));
    SetVector<Value> inputs(genericOp.getInputs().begin(),
                            genericOp.getInputs().end());

    assert(all_of(args,
                  [](Value arg) { return isa<MemRefType>(arg.getType()); }) &&
           "All generic arguments should be memref.");

    for (auto arg : args) {
      // Convert memref def, allocOp
      auto allocOp = dyn_cast<memref::AllocOp>(arg.getDefiningOp());
      assert(allocOp && "Expecting allocOp as the memref definition");
      auto localTensorOp = convertAllocOp(allocOp, tpipeOp.getTpipe());
      assert(localTensorOp &&
             "Convert memref alloc to ascendc alloctensor failed");
      if (inputs.contains(arg))
        genericNewInputs.push_back(localTensorOp.getLocalTensor());
      else
        genericNewOutputs.push_back(localTensorOp.getLocalTensor());
      // Convert memref users, currently supports: copyOp, deallocOp.
      for (auto *argUse : arg.getUsers()) {
        if (dyn_cast<linalg::GenericOp>(argUse))
          continue;
        if (auto copyOp = dyn_cast<memref::CopyOp>(argUse)) {
          bool gm2ub = arg == copyOp.getTarget();
          convertCopyOp(copyOp, gm2ub, tpipeOp, localTensorOp, gmAddr);
        } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(argUse)) {
          convertDeallocOp(deallocOp, localTensorOp);
        } else {
          llvm_unreachable("Unexpected Memref Usage");
          return;
        }
      }
    }

    // Convert generic op.
    convertGenericOp(genericOp, genericNewInputs, genericNewOutputs);
  }

  // Dealloc will also be inserted for memref on GM addrspace.
  SmallVector<Operation *> deallocOpsToDelete;
  func.walk([&](memref::DeallocOp op) { deallocOpsToDelete.push_back(op); });
  for (Operation *op : deallocOpsToDelete) {
    op->erase();
  }

  // Clean up alloc ops.
  SmallVector<Operation *> allocOpsToDelete;
  func.walk([&](memref::AllocOp op) { allocOpsToDelete.push_back(op); });
  for (Operation *op : allocOpsToDelete) {
    op->erase();
  }
}

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createConvertToAscendCPass() {
  return std::make_unique<ConvertToAscendCPass>();
}
