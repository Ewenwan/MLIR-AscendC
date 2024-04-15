//===- Promotion.cpp - Implementation of ascendc Promotion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ascendc promotion reuseing linlag promotion.
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir-ascendc/Dialect/AscendC/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

using namespace mlir;
using namespace mlir::linalg;

LogicalResult mlir::ascendc::promoteSubviewsAscendCPrecondition(
    Operation *op, OpBuilder &b, LinalgPromotionOptions options) {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  // Transformation applies to buffers only.
  if (!linalgOp || !linalgOp.hasPureBufferSemantics())
    return failure();

  LogicalResult foundSV = failure();
  // Check that at least one of the requested operands is indeed a subview.
  for (OpOperand &opOperand : linalgOp->getOpOperands()) {
    auto sv =
        isa_and_nonnull<memref::SubViewOp>(opOperand.get().getDefiningOp());
    if (sv)
      foundSV = success();
    else if (!sv &&
             (!options.operandsToPromote ||
              options.operandsToPromote->count(opOperand.getOperandNumber()))) {
      // we want to promote this opernad, but we found no subview, we create a
      // full sized subview before the op and use it
      Location loc = b.getUnknownLoc();
      Value operandVal = opOperand.get();
      MemRefType valTy = cast<MemRefType>(operandVal.getType());
      ArrayRef<int64_t> shape = valTy.getShape();
      SmallVector<Value> offsets, sizes, strides;

      // insert dimOp/Constant/subview ops as before as possible
      if (isa<BlockArgument>(operandVal))
        b.setInsertionPointToStart(operandVal.getParentBlock());
      else
        b.setInsertionPointAfter(operandVal.getDefiningOp());
      Value constZero = b.create<arith::ConstantIndexOp>(loc, 0);
      Value constOne = b.create<arith::ConstantIndexOp>(loc, 1);

      for (unsigned i = 0; i < shape.size(); i++) {
        if (ShapedType::isDynamic(shape[i])) {
          // dynamic shape
          Value dynDim = b.create<memref::DimOp>(loc, operandVal, i);
          sizes.push_back(dynDim);
        } else {
          Value constDim = b.create<arith::ConstantIndexOp>(loc, shape[i]);
          sizes.push_back(constDim);
        }
        offsets.push_back(constZero);
        strides.push_back(constOne);
      }
      // make the new subview
      Value newSV =
          b.create<memref::SubViewOp>(loc, operandVal, offsets, sizes, strides);
      // only replace the usage of the original memref for other LinalgOps
      // because those LinalgOp can have the same problem (want to promote but
      // no subview) as the current one
      operandVal.replaceUsesWithIf(newSV, [](OpOperand &operand) -> bool {
        return isa<LinalgOp>(operand.getOwner());
      });
      foundSV = success();
    }
  }
  // TODO: Check all subviews requested are bound by a static constant.
  // TODO: Check that the total footprint fits within a given size.
  return foundSV;
}

// Allocate subview in the specified AscendC TPosition
static std::optional<Value>
allocateSubviewInAscendCTPosition(OpBuilder &builder, memref::SubViewOp subview,
                                  ArrayRef<Value> sizeBounds,
                                  ascendc::TPosition tposition) {
  OpBuilder::InsertionGuard guard(builder);

  func::FuncOp funcOp = subview->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return std::nullopt;

  // The subview size bounds are expected to be constant; they specify the shape
  // of the allocation.
  SmallVector<int64_t> shape;
  for (Value bound : sizeBounds) {
    APInt value;
    if (!matchPattern(bound, m_ConstantInt(&value)))
      return std::nullopt;
    shape.push_back(value.getSExtValue());
  }

  builder.setInsertionPoint(&funcOp.front(), funcOp.front().begin());
  auto type = MemRefType::get(
      shape, subview.getType().getElementType(), MemRefLayoutAttrInterface{},
      ascendc::TPositionAttr::get(builder.getContext(), tposition));
  Value buffer;
  if (tposition == ascendc::AscendCDialect::getVecinTPosition()) {
    buffer = builder.create<memref::AllocOp>(funcOp.getLoc(), type);
  } else {
    return std::nullopt;
  }
  return buffer;
}

/// Allocate the subview in the Davinci UB memory.
std::optional<Value>
mlir::ascendc::allocateAscendCVECIN(OpBuilder &builder,
                                    memref::SubViewOp subview,
                                    ArrayRef<Value> sizeBounds, DataLayout &) {
  return allocateSubviewInAscendCTPosition(
      builder, subview, sizeBounds,
      ascendc::AscendCDialect::getVecinTPosition());
}

/// Normal copy to between src and dst.
LogicalResult mlir::ascendc::copyToAscendCVECIN(OpBuilder &b, Value src,
                                                Value dst) {
  b.create<memref::CopyOp>(src.getLoc(), src, dst);
  return success();
}

/// Deallocate Davinci UB memory
LogicalResult mlir::ascendc::deallocateAscendCVECIN(OpBuilder &,
                                                    Value /*buffer*/) {
  return success();
}
