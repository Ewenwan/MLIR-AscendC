//===- InitAllTranslations.h - MLIR Translations Registration ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all translations
// in and out of MLIR to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_INITALLTRANSLATIONS_H
#define MLIRASCENDC_INITALLTRANSLATIONS_H

#include "mlir-ascendc/Target/Cpp/translateAscendCToCpp.h"

namespace mlir_ascendc {

void registerAscendCToCppTranslation();

inline void registerAllTranslations() {

  static bool initOnce = []() {
  // AscendC translation
    registerAscendCToCppTranslation();

    return true;
  }();
  (void)initOnce;
}

} // namespace mlir_ascendc

#endif // MLIRASCENDC_INITALLTRANSLATIONS_H
