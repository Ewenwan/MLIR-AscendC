//===- AscendC.td - AscendC dialect definitions -----------*- tablegen -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR AscendC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDC_BASE
#define MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDC_BASE

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// Dialect Declaration
//===----------------------------------------------------------------------===//

def AscendC_Dialect : Dialect {
  let name = "ascendc";
  let cppNamespace = "::mlir::ascendc";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let summary = [{
    This dialect provides an representation of AscendC language operations.
  }];

  let extraClassDeclaration = [{
    /// Returns the numeric value used to identify the TPosition.
    static TPosition getVecinTPosition() { return TPosition::VECIN; }
  }];
}

#endif // MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDC_BASE
