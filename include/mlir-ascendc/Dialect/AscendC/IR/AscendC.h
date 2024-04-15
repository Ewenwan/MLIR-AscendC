//===----------- AscendC.h - MLIR AscendC Dialect ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AscendC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDC_H
#define MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDC_H

#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

// generated enum declaration
#include "mlir-ascendc/Dialect/AscendC/IR/AscendCEnums.h.inc"

// generated attribute declarations
#define GET_ATTRDEF_CLASSES
#include "mlir-ascendc/Dialect/AscendC/IR/AscendCAttributes.h.inc"

// generated dialect declaration
#include "mlir-ascendc/Dialect/AscendC/IR/AscendCDialect.h.inc"

// generated type declarations
#define GET_TYPEDEF_CLASSES
#include "mlir-ascendc/Dialect/AscendC/IR/AscendCTypes.h.inc"

// generated operation declarations
#define GET_OP_CLASSES
#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h.inc"

#endif // MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDC_H
