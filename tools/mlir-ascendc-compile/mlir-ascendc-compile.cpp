//===- mlir-ascendc-compile.cpp - MLIR compiler targetting ascendc
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Options.h"
#include "PassPipelines.h"
#include "mlir-ascendc/InitAllDialects.h"
#include "mlir-ascendc/InitAllExtensions.h"
#include "mlir-ascendc/InitAllPasses.h"
#include "mlir-ascendc/InitAllTranslations.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <experimental/filesystem>
#include <fstream>

#define DEBUG_TYPE "mlir-ascendc-compile"

using namespace mlir;
using namespace ascendc_compiler;

namespace {
// Helper method to check if any of the options provided signify that the user
// want to print the IR.
bool hasAnyPrintingOptions(const Options &options) {
  return options.printAfterAll || options.printBeforeAll ||
         options.printAfter.hasAnyOccurrences() ||
         options.printBefore.hasAnyOccurrences() || options.printAfterChange ||
         options.printModuleScope;
}

static LogicalResult convertToCpp(ModuleOp module, const Options &options) {
  std::string moduleStr;
  llvm::raw_string_ostream ss(moduleStr);

  auto cppTrans =
      mlir::translateAscendCToCpp(module, ss, options.toggleAscendCInfo);
  if (cppTrans.failed()) {
    LLVM_DEBUG(llvm::dbgs() << "MLIR Cpp Translation failed\n");
    return failure();
  }

  std::string name;
  if (!std::string(options.outputFile).empty())
    name = std::experimental::filesystem::path(std::string(options.outputFile));
  else {
    name = std::experimental::filesystem::path(std::string(options.inputFile))
               .stem();
    name += "_ascendc.cpp";
  }
  std::ofstream output(name);
  output << moduleStr;
  output.close();
  return success();
}

LogicalResult compile(MLIRContext &ctx, ModuleOp &moduleOp,
                      const Options &options) {
  PassManager modulePM(&ctx, ModuleOp::getOperationName(),
                       OpPassManager::Nesting::Implicit);
  addAscendCPipeline(modulePM, options);

  // Handle print-before.
  std::function<bool(Pass *, Operation *)> shouldPrintBeforePass;
  if (options.printBeforeAll) {
    // If we are printing before all, then just return true for the filter.
    shouldPrintBeforePass = [](Pass *, Operation *) { return true; };
  } else if (options.printBefore.hasAnyOccurrences()) {
    // Otherwise if there are specific passes to print before, then check to see
    // if the pass info for the current pass is included in the list.
    shouldPrintBeforePass = [&](Pass *pass, Operation *) {
      auto *passInfo = pass->lookupPassInfo();
      return passInfo && options.printBefore.contains(passInfo);
    };
  }

  // Handle print-after.
  std::function<bool(Pass *, Operation *)> shouldPrintAfterPass;
  if (options.printAfterAll) {
    // If we are printing after all, then just return true for the filter.
    shouldPrintAfterPass = [](Pass *, Operation *) { return true; };
  } else if (options.printAfter.hasAnyOccurrences()) {
    // Otherwise if there are specific passes to print after, then check to see
    // if the pass info for the current pass is included in the list.
    shouldPrintAfterPass = [&](Pass *pass, Operation *) {
      auto *passInfo = pass->lookupPassInfo();
      return passInfo && options.printAfter.contains(passInfo);
    };
  }

  if (hasAnyPrintingOptions(options)) {
    // enable IR Printing only if a print option was selected.
    modulePM.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                              options.printModuleScope,
                              options.printAfterChange,
                              /*printAfterOnlyOnFailure*/ false, llvm::errs());
  }

  // Run the optimization pipeline.
  if (failed(modulePM.run(moduleOp))) {
    LLVM_DEBUG(llvm::dbgs() << "Lowering failed\n");
    return failure();
  }
  return convertToCpp(moduleOp, options);
}
} // namespace

int main(int argc, char **argv) {
  // Register the dialects, passes, and translations used.
  // We should really only be registering the passes and dialects that we are
  // using.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::registerAllExtensions(registry);
  // AscendC Dialect specific registration
  mlir_ascendc::registerAllDialects(registry);
  mlir_ascendc::registerAllPasses();
  mlir_ascendc::registerAllExtensions(registry);
  mlir_ascendc::registerAllTranslations();

  // Handle command line options
  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "mlir-ascendc-compile\n");

  // Create the context
  MLIRContext context(registry);
  if (hasAnyPrintingOptions(options)) {
    // Disabling multi-threading so we can print IR
    context.disableMultithreading();
  }
  context.loadAllAvailableDialects();

  // Open the input file anf get the module.
  std::string errorMessage;
  auto file = openInputFile(options.inputFile, &errorMessage);
  if (!file) {
    llvm::errs() << "Error opening " << options.inputFile << ": "
                 << errorMessage << '\n';
    return EXIT_FAILURE;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  OwningOpRef<ModuleOp> moduleRef =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!moduleRef) {
    llvm::errs() << "Error opening " << options.inputFile << '\n';
    return EXIT_FAILURE;
  }
  ModuleOp module = *moduleRef;

  // Compile the module to LLVMIR.
  if (failed(compile(context, module, options)))
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
