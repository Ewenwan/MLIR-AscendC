//===- PassPipelines.cpp - Pass management for mlir-ascendc-compile -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassPipelines.h"
#include "Options.h"
#include "mlir-ascendc/Dialect/AscendC/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
template <typename Derived>
class OpPassWrapper : public PassWrapper<Derived, OperationPass<>> {};
class TestTransformDialectInterpreterPass
    : public mlir::transform::TransformInterpreterPassBase<
          TestTransformDialectInterpreterPass, OpPassWrapper> {
public:
  TestTransformDialectInterpreterPass(
      const TestTransformDialectInterpreterPass &pass)
      : TransformInterpreterPassBase(pass) {}
  TestTransformDialectInterpreterPass(const std::string &passedFileName)
      : TransformInterpreterPassBase() {
    transformFileName = passedFileName;
  }

  void runOnOperation() override {
    if (failed(transform::detail::interpreterBaseRunOnOperationImpl(
            getOperation(), getArgument(), getSharedTransformModule(),
            getTransformLibraryModule(), {}, {}, transformFileName,
            transformLibraryPaths, debugPayloadRootTag, debugTransformRootTag,
            {})))
      return signalPassFailure();
  }

  Option<std::string> transformFileName{*this, ""};
  Option<std::string> debugPayloadRootTag{*this, ""};
  Option<std::string> debugTransformRootTag{*this, ""};
  ListOption<std::string> transformLibraryPaths{*this, ""};
};
namespace ascendc_compiler {

void addAscendCPipeline(PassManager &pm, const Options &options) {
  //--------------------------------------------------------------------------//
  // Pre-Processing Passes
  //  - Prepares the IR for the upcoming passes.
  //--------------------------------------------------------------------------//
  pm.addPass(ascendc::createLegalizeFuncForAscendCPass());
  pm.addPass(createInlinerPass());
  pm.addPass(createLinalgInlineScalarOperandsPass());

  //--------------------------------------------------------------------------//
  // Promotion & Bufferization
  //--------------------------------------------------------------------------//
  if (!std::string(options.transformFile).empty())
    pm.addPass(std::make_unique<TestTransformDialectInterpreterPass>(
        (std::string)options.transformFile));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(ascendc::createAscendCBufferizationSimplificationPass());
  pm.addPass(memref::createExpandReallocPass());
  pm.addPass(bufferization::createBufferDeallocationPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(bufferization::createLowerDeallocationsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  //--------------------------------------------------------------------------//
  // Convert To AscendC
  //  - AscendC lowering and legalization
  //--------------------------------------------------------------------------//
  pm.addPass(ascendc::createAscendCJustificationPass());
  pm.addPass(ascendc::createConvertToAscendCPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(ascendc::createAscendCAutoCompletePass());
}

} // namespace ascendc_compiler
} // namespace mlir
