//===- AscendCConversionPasses.td - Conversion pass definition - tablegen -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_CONVERSION_ASCENDCCONVERSIONPASSES
#define MLIRASCENDC_CONVERSION_ASCENDCCONVERSIONPASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AscendCToEmitC
//===----------------------------------------------------------------------===//

def ConvertAscendCToEmitC : Pass<"convert-ascendc-to-emitc", "ModuleOp"> {
  let summary = "Lower AscendC to EmitC Dialect";
  let description =
      [{Converts AscendC Dialect to EmitC for translation to cpp.
        This is still WIP and may need modification with the upstream EmitC
        development. An AscendCCppEmitter has been forked and provided for
        current cpp transaltion.}];
  let constructor = "mlir::createConvertAscendCToEmitCPass()";
  let dependentDialects = [
    "emitc::EmitCDialect"
  ];
}

#endif // MLIRASCENDC_CONVERSION_ASCENDCCONVERSIONPASSES
