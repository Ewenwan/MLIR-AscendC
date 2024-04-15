//===- AscendCTransformOps.td - AscendC Transform Ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMOPS_ASCENDCTRANSFORMOPS
#define MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMOPS_ASCENDCTRANSFORMOPS

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

def AscendCPromoteOp : Op<Transform_Dialect, "ascendc.promote",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait,
    ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    This is the promotion op based on transform::PromoteOp specialized for
    AscendC operation. It provides a way to promote original illegal opeartions
    and supports promote to AscendC TPosition address space.

    For more information about promote op, please visit:
    https://mlir.llvm.org/docs/Dialects/Transform/#transformstructuredpromote-transformpromoteop
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       DefaultValuedAttr<I64ArrayAttr, "{}">:$operands_to_promote,
                       DefaultValuedAttr<BoolArrayAttr, "{}">:$use_full_tile_buffers,
                       UnitAttr:$use_full_tiles_by_default,
                       UnitAttr:$use_alloca,
                       UnitAttr:$copyout_only,
                       OptionalAttr<AnyAttr>:$memory_space,
                       OptionalAttr<DeviceMappingArrayAttr>:$mapping,
                       OptionalAttr<I64Attr>:$alignment);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat =
    "$target attr-dict `:`"
    "custom<SemiFunctionType>(type($target), type($transformed))";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // MLIRASCENDC_DIALECT_ASCENDC_TRANSFORMOPS_ASCENDCTRANSFORMOPS
