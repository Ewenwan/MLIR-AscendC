//===- InitAllPasses.h - MLIR Passes Registration ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_INITALLPASSES_H_
#define MLIRASCENDC_INITALLPASSES_H_

#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h"

namespace mlir_ascendc {

// This function may be called to register the ascendc-specific MLIR passes with
// the global registry.
inline void registerAllPasses() { mlir::ascendc::registerAscendCPasses(); }

} // namespace mlir_ascendc

#endif // MLIRASCENDC_INITALLPASSES_H_
