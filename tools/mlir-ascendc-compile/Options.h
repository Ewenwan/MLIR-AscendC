//===------------ Options.h - Options for mlir-ascendc-compile ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_TOOLS_MLIR_ASCENDC_COMPILE_OPTIONS_H
#define MLIRASCENDC_TOOLS_MLIR_ASCENDC_COMPILE_OPTIONS_H

#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace ascendc_compiler {
namespace cl = llvm::cl;

struct Options {
  cl::opt<std::string> inputFile{
      cl::Positional, cl::desc("the input .mlir file"), cl::init("")};
  cl::opt<std::string> outputFile{
      "o", cl::Positional, cl::desc("the output .cpp file"), cl::init("")};

  PassNameCLParser printBefore{"ascendc-print-ir-before",
                               "Print IR before specified passes"};
  PassNameCLParser printAfter{"ascendc-print-ir-after",
                              "Print IR after specified passes"};

  cl::opt<bool> printBeforeAll{"ascendc-print-ir-before-all",
                               cl::desc("Print IR before each pass"),
                               cl::init(false)};

  cl::opt<bool> printAfterAll{"ascendc-print-ir-after-all",
                              cl::desc("Print IR after each pass"),
                              cl::init(false)};

  cl::opt<bool> printAfterChange{
      "ascendc-print-ir-after-change",
      cl::desc(
          "When printing the IR after a pass, only print if the IR changed"),
      cl::init(false)};

  cl::opt<bool> printModuleScope{
      "ascendc-print-ir-module-scope",
      cl::desc("When printing IR for print-ir-[before|after]{-all} "
               "always print the top-level operation"),
      cl::init(false)};

  cl::opt<bool> toggleAscendCInfo{
      "remove-ascendc-info",
      cl::desc("check if the ascendc translation needs to add ascendc specific"
               "information"),
      cl::init(false)};

  cl::opt<std::string> transformFile{
      "transform-file", cl::Optional,
      cl::desc("transform template file for mlir"), cl::init("")};
};

} // namespace ascendc_compiler
} // namespace mlir

#endif // MLIRASCENDC_TOOLS_MLIR_ASCENDC_COMPILE_OPTIONS_H
