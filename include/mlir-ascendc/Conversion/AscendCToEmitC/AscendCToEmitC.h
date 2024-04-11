//===--- AscendCToEmitC.h - Convert from the AscendC dialect to EmitC -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_CONVERSION_ASCENDCTOEMITC_ASCENDCTOEMITC_H
#define MLIRASCENDC_CONVERSION_ASCENDCTOEMITC_ASCENDCTOEMITC_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

#define GEN_PASS_DECL_CONVERTASCENDCTOEMITC
#include "mlir-ascendc/Conversion/AscendCConversionPasses.h.inc"

class ModuleOp;
template <typename T> class OperationPass;
class RewritePatternSet;

namespace ascendc {
/// Collect the patterns to convert from the HIVM dialect to LLVM.
void populateAscendCToEmitCConversionPatterns(RewritePatternSet &patterns);
} // namespace ascendc

std::unique_ptr<OperationPass<ModuleOp>> createConvertAscendCToEmitCPass();

} // namespace mlir

#endif // MLIRASCENDC_CONVERSION_ASCENDCTOEMITC_ASCENDCTOEMITC_H
