//===- AscendC.td - AscendC dialect definitions -----------*- tablegen -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR AscendC Types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDCTYPES
#define MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDCTYPES

include "mlir-ascendc/Dialect/AscendC/IR/AscendCBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// Type Declarations
//===----------------------------------------------------------------------===//

class AscendC_Type<string typeName, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<AscendC_Dialect, typeName, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def AnyIntegerLike : TypeConstraint<Or<[SignlessIntegerLike.predicate,
    AnySignedInteger.predicate]>, "any integer">;
def AnyIntegerType : Type<AnyIntegerLike.predicate, "any integer">;

//===----------------------------------------------------------------------===//
// TPipe
//===----------------------------------------------------------------------===//

def AscendC_TPipe : AscendC_Type<"TPipe", "TPipe"> {
  let summary = "AscendC TPipe";

  let description = [{
    Tpipe is the overall memory management facility. TPipe is used to manage
    memory for TQueue and TBuf.}];
}

//===----------------------------------------------------------------------===//
// TQue & TBuf
//===----------------------------------------------------------------------===//

def AscendC_TQue : AscendC_Type<"TQue", "TQue"> {
  let summary = "AscendC TQue";

  let description = [{
    TQue is managed by TPipe and is used for buffer management, data communication
    and synchronization between different tasks. TQue used QuePosition (TPosition)
    to manage various level of physcial hardware: VECIN, VECCALC, VECOUT, A1, A2,
    B1, B2, CO1, CO2
  }];

  let parameters = (ins "::mlir::ascendc::TPosition":$tposition,
                        "unsigned":$bufNum);
  let assemblyFormat = "`<` $tposition `,` $bufNum `>`";
}

def AscendC_TBuf : AscendC_Type<"TBuf", "TBuf"> {
  let summary = "AscendC TBuf";

  let description = [{
    TBuf is managed by TPipe and is used for managing temporary buffer. Only a
    single piece of memory will be allocated for TBuf and memory allocated can
    only be used to compute while cannot take part in Enque or Deque. TBuf used
    TPosition to manage various level of physcial hardware: VECIN, VECCALC, VECOUT,
    A1, A2, B1, B2, CO1, CO2
  }];

  let parameters = (ins "::mlir::ascendc::TPosition":$tposition);
  let assemblyFormat = "`<` $tposition `>`";
}

//===----------------------------------------------------------------------===//
// GM_ADDR
//===----------------------------------------------------------------------===//

def AscendC_GMADDR : AscendC_Type<"GM_ADDR", "GM_ADDR"> {
  let summary = "AscendC GM_ADDR";

  let description = [{ GM_ADDR represents global memory address.}];
}

//===----------------------------------------------------------------------===//
// GlobalTensor & LocalTensor
//===----------------------------------------------------------------------===//

class AscendC_TensorType<string typeName, string typeMnemonic, list<Trait> traits = []>
    : AscendC_Type<typeName, typeMnemonic, !listconcat([ShapedTypeInterface], traits)> {

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType),
      [{return $_get(elementType.getContext(), shape, elementType);}]
    >
  ];
}

def AscendC_LocalTensor : AscendC_TensorType<"LocalTensor", "LocalTensor"> {
  let summary = "AscendC LocalTensor";
  let description = [{
    LocalTensor is used to store data inside local memory of AI core.
    Supported QuePosition: VECIN, VECOUT, A1, A2, B1, B2, CO1, CO2.
  }];
  let extraClassDeclaration = [{
    bool hasRank() const { return true; }

    LocalTensorType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                         Type elementType) const;
  }];
  let hasCustomAssemblyFormat = 1;
}

def AscendC_GlobalTensor : AscendC_TensorType<"GlobalTensor", "GlobalTensor"> {
  let summary = "AscendC GlobalTensor";
  let description = [{GlobalTensor is used to store data on global memory.}];
  let extraClassDeclaration = [{
    bool hasRank() const { return true; }
aln
    GlobalTensorType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                         Type elementType) const;
  }];
  let hasCustomAssemblyFormat = 1;
}

def AscendC_AnyTensorType : AnyTypeOf<[
  AscendC_LocalTensor,
  AscendC_GlobalTensor
], "AscendC Tensor Types">;

#endif
