//===- AscendCBufferizationSimplification.cpp - Code to simplify pattern --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass simplifies a pattern that a memref is allocated and only used for a
// copy in and immediately followed by a copy out. This pass replaces these two
// copyops with one and erase the extra memref allocation.
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_ASCENDCBUFFERIZATIONSIMPLIFICATION
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace ascendc;

namespace {

static Value getBaseMemRef(Value v) {
  while (Operation *defOp = v.getDefiningOp()) {
    if (memref::SubViewOp subview = dyn_cast<memref::SubViewOp>(defOp))
      v = subview.getSource();
    else {
      break;
    }
  }
  return v;
}

// Helper method to check if the src memref can be reused in place of the tar
// MemRef (just looking at the type information).
static bool canReuseMemRef(Value src, Value tar) {
  // Cannot replace a MemRef with itself
  if (src == tar)
    return false;
  MemRefType srcTy = cast<MemRefType>(src.getType());
  MemRefType tarTy = cast<MemRefType>(tar.getType());
  // They must be the same shape.
  if (!srcTy.getShape().equals(tarTy.getShape()))
    return false;
  // They must be the same element type.
  if (srcTy.getElementType() != tarTy.getElementType())
    return false;
  // They must have the same memory space.
  // NOTE: currently allow all AscendC TPosition Memspace.
  if (isa<TPositionAttr>(srcTy.getMemorySpace()) &&
      isa<TPositionAttr>(tarTy.getMemorySpace()))
    return true;
  if (srcTy.getMemorySpace() != tarTy.getMemorySpace())
    return false;
  return true;
}

struct AscendCBufferizationSimplificationPass
    : public ascendc::impl::AscendCBufferizationSimplificationBase<
          AscendCBufferizationSimplificationPass> {
  void runOnOperation() override;
};

void AscendCBufferizationSimplificationPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Walk all the ops in the MemRefDialect
  SetVector<Operation *> opsToDelete;
  SetVector<memref::AllocOp> allocOps;

  func.walk([&](memref::AllocOp op) { allocOps.insert(op); });

  for (Operation *allocOp : allocOps) {
    // Match the copies in and out
    memref::CopyOp copyIn;
    memref::CopyOp copyOut;
    for (Operation *user : allocOp->getUsers()) {
      if (dyn_cast<memref::SubViewOp>(user))
        continue;
      memref::CopyOp copyOp = dyn_cast<memref::CopyOp>(user);
      if (!copyOp) {
        copyIn = nullptr;
        copyOut = nullptr;
        break;
      }
      if (copyOp.getSource() == allocOp->getResult(0)) {
        if (copyOut) {
          copyOut = nullptr;
          break;
        }
        copyOut = copyOp;
        continue;
      }
      if (copyIn) {
        copyIn = nullptr;
        break;
      }
      copyIn = copyOp;
    }
    // If only a single copy in and copy out exist, remove these two copies and
    // replace with a new copy out from the src of copyin to dst of copyout.
    if (copyIn && copyOut) {
      Value srcBaseMemRef = getBaseMemRef(copyIn.getSource());
      Value dstBaseMemRef = getBaseMemRef(copyOut.getTarget());
      if (!canReuseMemRef(srcBaseMemRef, dstBaseMemRef))
        continue;
      opsToDelete.insert(copyIn);
      opsToDelete.insert(copyOut);

      // insert new copy here
      OpBuilder builder(copyOut);
      builder.create<memref::CopyOp>(copyOut->getLoc(), srcBaseMemRef,
                                     dstBaseMemRef);
    }
  }

  // Erase all ops marked for death.
  for (Operation *op : opsToDelete)
    op->erase();

  // Clean up any extra allocation.
  SetVector<Operation *> allocationsToCheck;
  func.walk([&](Operation *op) {
    if (isa<memref::AllocOp>(op))
      allocationsToCheck.insert(op);
  });
  for (Operation *alloc : allocationsToCheck) {
    if (alloc->use_empty()) {
      alloc->erase();
      continue;
    }
  }
}
} // namespace

std::unique_ptr<Pass>
mlir::ascendc::createAscendCBufferizationSimplificationPass() {
  return std::make_unique<AscendCBufferizationSimplificationPass>();
}
