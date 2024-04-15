//===- AscendCTransformOps.cpp - Implementation of AscendC transform ops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Dialect/AscendC/TransformOps/AscendCTransformOps.h"
#include "mlir-ascendc/Dialect/AscendC/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// PromoteOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::AscendCPromoteOp::applyToOne(
    transform::TransformRewriter &rewriter, LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  LinalgPromotionOptions promotionOptions;
  if (!getOperandsToPromote().empty())
    promotionOptions = promotionOptions.setOperandsToPromote(
        extractFromIntegerArrayAttr<int64_t>(getOperandsToPromote()));
  if (getUseFullTilesByDefault())
    promotionOptions = promotionOptions.setUseFullTileBuffersByDefault(
        getUseFullTilesByDefault());
  if (getUseAlloca())
    promotionOptions = promotionOptions.setUseAlloca(getUseAlloca());
  if (!getUseFullTileBuffers().empty())
    promotionOptions = promotionOptions.setUseFullTileBuffers(
        llvm::to_vector(getUseFullTileBuffers().getAsValueRange<BoolAttr>()));
  if (getAlignment().has_value())
    promotionOptions = promotionOptions.setAlignment(*getAlignment());
  if (getMemorySpace().has_value())
    promotionOptions = promotionOptions.setMemorySpace(*getMemorySpace());

  if (getMapping().has_value()) {
    // The mapping should only contain an element
    auto mapping = *getMapping();
    if (mapping.size() > 1)
      return emitDefaultDefiniteFailure(target);

    auto ascendcTposition = dyn_cast<ascendc::TPositionAttr>(mapping[0]);

    if (ascendcTposition && ascendcTposition.getValue() ==
                                ascendc::AscendCDialect::getVecinTPosition()) {
      promotionOptions =
          promotionOptions
              .setAllocationDeallocationFns(ascendc::allocateAscendCVECIN,
                                            ascendc::deallocateAscendCVECIN)
              .setCopyInOutFns(ascendc::copyToAscendCVECIN,
                               ascendc::copyToAscendCVECIN)
              .setUseFullTileBuffers({false, false});
    } else {
      return emitDefaultDefiniteFailure(target);
    }
  }
  if (failed(ascendc::promoteSubviewsAscendCPrecondition(target, rewriter,
                                                         promotionOptions)))
    return emitDefaultDefiniteFailure(target);
  rewriter.setInsertionPoint(target);
  FailureOr<LinalgOp> res = promoteSubViews(rewriter, target, promotionOptions);
  if (failed(res))
    return emitDefaultDefiniteFailure(target);
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "mlir-ascendc/Dialect/AscendC/TransformOps/AscendCTransformOps.cpp.inc"
