//===---- LegalizeFuncForAscendC.cpp - Legalize func usage for AscendC ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file legalizes the function for AscendC target.
// 1) It creates a new function that turns the original operands to GM_ADDR
//    types while also keeps the original function. Therefore, this pass
//    currently expects the user to run the inliner after running this pass.
//
// 2) Converts the tensor operands and results into memref allocations with
//    bufferization.to_tensor op.
//
// Note: This pass is adapted from a pass by Alex Singer in a previous project.
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_LEGALIZEFUNCFORASCENDC
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace ascendc;

namespace {

struct LegalizeFuncForAscendCPass
    : public ascendc::impl::LegalizeFuncForAscendCBase<
          LegalizeFuncForAscendCPass> {
  void runOnOperation() override;
};

// Checks if the given FuncOp is legal.
static bool isFuncLegalForAscendC(func::FuncOp func) {
  // Cannot have any results
  if (func.getNumResults() > 0)
    return false;
  // Arguments cannot be tensors.
  ArrayRef<Type> argTys = func.getArgumentTypes();
  return all_of(argTys, [](Type argTy) { return !isa<TensorType>(argTy); });
}

// Converts TensorTypes into the corresponding MemRefType in address space 1.
static Type convertArgumentTyMemRefTy(Type ty) {
  TensorType tensorTy = dyn_cast<TensorType>(ty);
  if (!tensorTy)
    return ty;
  Attribute memspace =
      ascendc::TPositionAttr::get(ty.getContext(), TPosition::GM);
  AffineMap map = {};
  return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), map,
                         memspace);
}

void LegalizeFuncForAscendCPass::runOnOperation() {
  ModuleOp module = getOperation();
  // Collect the FuncOps that need to be legalized.
  SmallVector<func::FuncOp> funcsToLegalize;
  module.walk([&](func::FuncOp func) {
    if (!isFuncLegalForAscendC(func))
      funcsToLegalize.push_back(func);
  });
  // Legalize the FuncOps by creating a new function with the correct prototype
  // that will call the original function.
  for (func::FuncOp func : funcsToLegalize) {
    // Get the prototype of the new function, converting tensors to memrefs and
    // turning results into arguments.
    ArrayRef<Type> argTys = func.getArgumentTypes();
    auto origArgs = func.getArguments();
    ArrayRef<Type> resTys = func.getResultTypes();
    size_t numOrigArgs = argTys.size();
    SmallVector<Type> newArgTys;
    newArgTys.reserve(numOrigArgs + resTys.size());
    for (Type argTy : argTys)
      newArgTys.push_back(convertArgumentTyMemRefTy(argTy));
    for (Type resTy : resTys)
      newArgTys.push_back(convertArgumentTyMemRefTy(resTy));

    // Create fucntion with gm_addr as arguments.
    SmallVector<Type> newGMAddrArgs(newArgTys.size(),
                                    GM_ADDRType::get(&getContext()));
    FunctionType newFuncTy =
        FunctionType::get(&getContext(), newGMAddrArgs, {});

    // Build the new FuncOp, giving it the name of the original FuncOp and
    // renaming the original.
    OpBuilder builder(func);
    Location loc = func.getLoc();
    StringRef funcName = func.getName();
    auto newFunc = builder.create<func::FuncOp>(loc, funcName, newFuncTy);
    func.setName(funcName.str() + "_original");

    // Set the body of the new FuncOp. Alloc the memref for each tensor and add
    // conversions to convert the memref args to tensors, add the call to the
    // original FuncOp, and add copy from the tensor result to memref.
    Block *newFuncBody = newFunc.addEntryBlock();
    builder.setInsertionPointToEnd(newFuncBody);

    SmallVector<Value> callOpArguments;
    callOpArguments.reserve(numOrigArgs);
    for (size_t i = 0; i < numOrigArgs; i++) {
      if (!isa<TensorType>(argTys[i])) {
        callOpArguments.push_back(origArgs[i]);
        continue;
      }
      auto memrefType = dyn_cast<MemRefType>(newArgTys[i]);
      Value argMemref = builder.create<memref::AllocOp>(loc, memrefType);

      // In order for one-shot bufferize to work on this, the to tensor ops need
      // to be restricted (meaning that the memref does not alias) and writable.
      UnitAttr restr = UnitAttr::get(&getContext());
      UnitAttr writable = UnitAttr::get(&getContext());
      Value newTensor = builder.create<bufferization::ToTensorOp>(
          loc, argMemref, restr, writable);
      callOpArguments.push_back(newTensor);
    }

    auto callOp = builder.create<func::CallOp>(loc, func, callOpArguments);
    assert(resTys.size() <= 1 && "TODO: handle case with 2+ results.");
    if (callOp.getNumResults() == 1) {
      Value callResult = callOp.getResult(0);
      auto memrefType = dyn_cast<MemRefType>(newArgTys[numOrigArgs]);
      Value newResultArg = builder.create<memref::AllocOp>(loc, memrefType);
      auto materializeOp =
          builder.create<bufferization::MaterializeInDestinationOp>(
              loc, callResult, newResultArg);
      materializeOp.setWritable(true);
    }
    builder.create<func::ReturnOp>(loc);
    // Set the original FuncOp to private visibility so that it can be removed
    // from the Inliner.
    func.setPrivate();
  }

  // Remove the ml_program.global which cannot be lowered at the moment.
  SmallVector<ml_program::GlobalOp> glops;
  module.walk([&](ml_program::GlobalOp glop) { glops.push_back(glop); });
  for (ml_program::GlobalOp glop : glops)
    glop.erase();
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::ascendc::createLegalizeFuncForAscendCPass() {
  return std::make_unique<LegalizeFuncForAscendCPass>();
}
