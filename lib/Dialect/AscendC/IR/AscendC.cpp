//===-------------- AscendC.cpp - AscendC ops implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ascendc;

#include "mlir-ascendc/Dialect/AscendC/IR/AscendCDialect.cpp.inc"

static TPosition getTensorTposition(Value tensor) {
  if (dyn_cast<GlobalTensorType>(tensor.getType()))
    return TPosition::GM;
  if (auto allocTensor = dyn_cast<AllocTensorOp>(tensor.getDefiningOp()))
    return allocTensor.getQueue().getType().getTposition();
  if (auto dequeOp = dyn_cast<DequeOp>(tensor.getDefiningOp()))
    return dequeOp.getQueue().getType().getTposition();
  return TPosition::MAX;
}

LogicalResult DataCopyOp::verify() {
  TPosition dstPosition = getTensorTposition(getDstTensor());
  TPosition srcPosition = getTensorTposition(getSrcTensor());

  // Unfound TPosition
  if (dstPosition == TPosition::MAX || srcPosition == TPosition::MAX) {
    return emitOpError("unfound TPosition");
  }

  // Supported: GM->A1, GM->B1, GM->VECIN
  if (srcPosition == TPosition::GM) {
    if (dstPosition == TPosition::A1 || dstPosition == TPosition::B1 ||
        dstPosition == TPosition::VECIN) {
      return success();
    }
  }
  // Supported: CO2->GM, VECOUT->GM
  if (dstPosition == TPosition::GM) {
    if (srcPosition == TPosition::CO2 || srcPosition == TPosition::VECOUT) {
      return success();
    }
  }
  // Supported: CO1->CO2, VECIN->VECOUT
  if (srcPosition == TPosition::CO1 && dstPosition == TPosition::CO2)
    return success();
  if (srcPosition == TPosition::VECIN && dstPosition == TPosition::VECOUT)
    return success();

  return emitOpError("unsupported data copy path");
}

//===----------------------------------------------------------------------===//
// AscendCDialect
//===----------------------------------------------------------------------===//

void ascendc::AscendCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-ascendc/Dialect/AscendC/IR/AscendCTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-ascendc/Dialect/AscendC/IR/AscendCAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AscendC Ops
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.cpp.inc"

//===----------------------------------------------------------------------===//
// AscendC Enums
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/IR/AscendCEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// AscendC Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir-ascendc/Dialect/AscendC/IR/AscendCAttributes.cpp.inc"

//----------------------------------------------------------------------------//
// AscendC Device Mapping Attributes
//----------------------------------------------------------------------------//

int64_t TPositionAttr::getMappingId() const {
  if (getValue() == ascendc::TPosition::VECIN ||
      getValue() == ascendc::TPosition::VECCALC ||
      getValue() == ascendc::TPosition::VECOUT)
    return static_cast<int64_t>(ascendc::TPosition::VECIN);
  return static_cast<int64_t>(getValue());
}

bool TPositionAttr::isLinearMapping() const {
  llvm_unreachable("TPositionAttr does not support linear mapping");
}

int64_t TPositionAttr::getRelativeIndex() const {
  llvm_unreachable("TPositionAttr does not support relative index");
}

//===----------------------------------------------------------------------===//
// AscendC Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir-ascendc/Dialect/AscendC/IR/AscendCTypes.cpp.inc"

//----------------------------------------------------------------------------//
// AscendC BaseTensor Types
//----------------------------------------------------------------------------//

Type parseTensorType(AsmParser &parser, bool localTensor) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t> sizes;
  if (parser.parseOptionalStar()) {
    if (parser.parseDimensionList(sizes, true))
      return Type();
  } else if (parser.parseXInDimensionList()) {
    return Type();
  }
  Type elementType;
  if (parser.parseType(elementType))
    return Type();

  if (parser.parseGreater())
    return Type();

  if (localTensor)
    return LocalTensorType::get(sizes, elementType);
  return GlobalTensorType::get(sizes, elementType);
}

static void printTensorType(AsmPrinter &printer, ArrayRef<int64_t> tensorShape,
                            Type elementType) {
  if (tensorShape.empty() && !elementType)
    return;
  printer << "<";
  if (!tensorShape.empty()) {
    for (auto elementSize : tensorShape) {
      if (elementSize < 0)
        printer << "?";
      else
        printer << elementSize;
      printer << "x";
    }
  } else {
    printer << "*x";
  }
  printer.printType(elementType);
  printer << ">";
}

//----------------------------------------------------------------------------//
// LocalTensor Type
//----------------------------------------------------------------------------//

Type LocalTensorType::parse(AsmParser &parser) {
  return parseTensorType(parser, /*localTensor=*/true);
}

void LocalTensorType::print(AsmPrinter &printer) const {
  printTensorType(printer, getShape(), getElementType());
}

LocalTensorType
LocalTensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                           Type elementType) const {
  return LocalTensorType::get(shape.value_or(getShape()), elementType);
}

//----------------------------------------------------------------------------//
// GlobalTensor Type
//----------------------------------------------------------------------------//

Type GlobalTensorType::parse(AsmParser &parser) {
  return parseTensorType(parser, /*localTensor=*/false);
}

void GlobalTensorType::print(AsmPrinter &printer) const {
  printTensorType(printer, getShape(), getElementType());
}

GlobalTensorType
GlobalTensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                            Type elementType) const {
  return GlobalTensorType::get(shape.value_or(getShape()), elementType);
}
