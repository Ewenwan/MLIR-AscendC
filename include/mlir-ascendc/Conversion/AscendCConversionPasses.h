//=- AscendCConversionPasses.h - Conversion Pass Creation and Registration -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_CONVERSION_ASCENDCCONVERSIONPASSES_H
#define MLIRASCENDC_CONVERSION_ASCENDCCONVERSIONPASSES_H

#include "mlir-ascendc/Conversion/AscendCToEmitC/AscendCToEmitC.h"
#include "mlir/Pass/Pass.h"

namespace mlir_ascendc {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mlir-ascendc/Conversion/AscendCConversionPasses.h.inc"

} // namespace mlir_ascendc

#endif // MLIRASCENDC_CONVERSION_ASCENDCCONVERSIONPASSES_H
