//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir-ascendc/Target/Cpp/translateAscendCToCpp.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace mlir_ascendc {

//===----------------------------------------------------------------------===//
// Cpp registration
//===----------------------------------------------------------------------===//

void registerAscendCToCppTranslation() {
  static llvm::cl::opt<bool> toggleAscendCSpecifics(
      "remove-ascendc-specifics",
      llvm::cl::desc("Remove info automatically added specific for AscendC"),
      llvm::cl::init(false));

  TranslateFromMLIRRegistration registration(
      "ascendc-to-cpp", "translate ascendc mlir to cpp",
      [](mlir::Operation *op, llvm::raw_ostream &output) {
        return mlir::translateAscendCToCpp(op, output, toggleAscendCSpecifics);
      },
      [](mlir::DialectRegistry &registry) {
        // clang-format off
        registry.insert<arith::ArithDialect,
                        ascendc::AscendCDialect,
                        func::FuncDialect,
                        scf::SCFDialect>();
        // clang-format on
      });
}

} // namespace mlir_ascendc
