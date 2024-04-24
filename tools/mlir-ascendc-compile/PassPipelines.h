//===--- PassPipelines.h - Pass management for mlir-ascendc-compile -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_TOOLS_MLIR_ASCENDC_COMPILE_PASSPIPELINES_H
#define MLIRASCENDC_TOOLS_MLIR_ASCENDC_COMPILE_PASSPIPELINES_H

#include "llvm/ADT/StringRef.h"
namespace mlir {

class PassManager;
class DataLayout;

namespace ascendc_compiler {
struct Options;

void addAscendCPipeline(PassManager &pm, const Options &options);

} // namespace ascendc_compiler
} // namespace mlir

#endif // MLIRASCENDC_TOOLS_MLIR_ASCENDC_COMPILE_PASSPIPELINES_H
