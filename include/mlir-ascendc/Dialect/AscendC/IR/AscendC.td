//===- AscendC.td - AscendC dialect definitions -----------*- tablegen -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR AscendC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDC
#define MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDC

include "mlir/IR/OpBase.td"
include "mlir-ascendc/Dialect/AscendC/IR/AscendCBase.td"
include "mlir-ascendc/Dialect/AscendC/IR/AscendCTypes.td"
include "mlir-ascendc/Dialect/AscendC/IR/AscendCAttributes.td"

//===----------------------------------------------------------------------===//
// Operation Declaration
//===----------------------------------------------------------------------===//

class AscendC_Op<string mnemonic, list<Trait> traits = []>
    : Op<AscendC_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Memory Management and Synchronization
//===----------------------------------------------------------------------===//

def CreatePipeOp : AscendC_Op<"create_pipe"> {
    let summary = "AscendC op to create pipe";
    let description = [{
        This is the op to create a TPipe, served as a constructor
        of TPipe in the Ascend C language.
    }];

    let results = (outs AscendC_TPipe: $tpipe);

    let assemblyFormat = "attr-dict `:` qualified(type($tpipe))";
}

def CreateQueueOp : AscendC_Op<"create_queue"> {
    let summary = "AscendC op to get queue from pipe";
    let description = [{
        This is the op to create TQue, served as a constructor of TQue
        combined with InitBuffer in AscendC language. TQue is managed
        by TPipe and $tile_length is the length of each buffer inside
        the queue.
    }];

    let arguments = (ins AscendC_TPipe: $tpipe,
                         I32: $tile_length);

    let results = (outs AscendC_TQue: $queue);

    let assemblyFormat =
      " $tpipe `,` $tile_length attr-dict `:` qualified(type($tpipe))"
      " `->` qualified(type($queue))";
}

def CreateTBufOp : AscendC_Op<"create_tbuf"> {
    let summary = "AscendC op to get queue from pipe";
    let description = [{
        This is the op to create TBuf, served as a constructor of Tbuf
        combined with InitBuffer in AscendC language. TBuf is managed
        by TPipe and $tile_length is the length of the buffer.
    }];

    let arguments = (ins AscendC_TPipe: $tpipe,
                         I32: $tile_length);

    let results = (outs AscendC_TBuf: $tbuf);

    let assemblyFormat =
      " $tpipe `,` $tile_length attr-dict `:` qualified(type($tpipe))"
      " `->` qualified(type($tbuf))";
}

def CreateGlobalTensorOp : AscendC_Op<"create_global_tensor"> {
    let summary = "AscendC op to create Global Tensor";
    let description = [{
        This is the op to create Global Tensor, served as a constructor of
        GlobalTensor combined with setGlobalBuffer in AscendC language. It
        takes $gm_address as initial address with $offset and $block_len.
        $block_len is the length of each block and $offset is calculated
        by (block_id * block_len).
    }];

    let arguments = (ins AscendC_GMADDR: $gm_address,
                         Optional<I32>: $offset,
                         I32: $block_len);

    let results = (outs AscendC_GlobalTensor: $globalTensor);

    let assemblyFormat =
      "$gm_address `,` $block_len (`,` $offset^)?"
      "attr-dict `:` qualified(type($gm_address)) "
      "`->` qualified(type($globalTensor))";
}

def TBufGetOp : AscendC_Op<"get"> {
    let summary = "AscendC op to get temp buffer";
    let description = [{
        Get the tensor from TBuf with specific size or the full size of the TBuf
    }];

    let arguments = (ins AscendC_TBuf: $tbuf,
                         Optional<I32>:$size);

    let results = (outs AscendC_LocalTensor: $localTensor);

    let assemblyFormat =
      "`(` $tbuf (`,` $size^)? `)` attr-dict `:` qualified(type($tbuf))"
      " `->` qualified(type($localTensor))";
}

def AllocTensorOp : AscendC_Op<"alloc_tensor"> {
    let summary = "AscendC AllocTensor Op";
    let description = [{
        Allocate the tensor from the queue, size of the tensor is the
        tile_length specified when creating the queue.
    }];

    let arguments = (ins AscendC_TQue: $queue,
                         Optional<I32>: $allocSize);
                        // TODO: OptionalAttr<I64ArrayAttr>

    let results = (outs AscendC_LocalTensor: $localTensor);

    let assemblyFormat =
      "`(` $queue (`,` $allocSize^)? `)` attr-dict `:` qualified(type($queue))"
      " `->` qualified(type($localTensor))";

    let extraClassDeclaration = [{
    LocalTensorType getLocalTensorType() {
      return ::llvm::cast<LocalTensorType>(getLocalTensor().getType());
    }
    }];
}

def FreeTensorOp : AscendC_Op<"free_tensor"> {
    let summary = "AscendC FreeTensor Op";
    let description = [{
        Free certain tensor in the queue for further usage.
    }];

    let arguments = (ins AscendC_TQue: $queue,
                         AscendC_LocalTensor: $localTensor);

    let assemblyFormat =
      " `(` $queue `,` $localTensor `)` attr-dict `:` qualified(type($localTensor))"
      "`from` qualified(type($queue))";
}

def EnqueOp : AscendC_Op<"enque"> {
    let summary = "AscendC Enque Op";
    let description = [{
        Push the local tensor into the corresponding queue.
    }];

    let arguments = (ins AscendC_TQue: $queue,
                         AscendC_LocalTensor: $localTensor);

    let assemblyFormat =
      " $localTensor `,` $queue attr-dict `:` qualified(type($localTensor))"
      " `to` qualified(type($queue))";
}

def DequeOp : AscendC_Op<"deque"> {
    let summary = "AscendC Deque Op";
    let description = [{
        Pop tensor from the corresponding queue for further operations.
    }];

    let arguments = (ins AscendC_TQue: $queue);

    let results = (outs AscendC_LocalTensor: $localTensor);

    let assemblyFormat =
      "`(` $queue `)` attr-dict `:` qualified(type($queue)) `->`"
      " qualified(type($localTensor))";
}

//===----------------------------------------------------------------------===//
// Data Processing
//===----------------------------------------------------------------------===//

def DataCopyOp : AscendC_Op<"data_copy",
    [AttrSizedOperandSegments]> {
    let summary = "AscendC Data Copy Op";
    let description = [{
        Coppy data from different positions. Supported data movement:
        GM->A1. GM->B1, CO1->CO2, CO2->GM, GM->VECIN, VECIN->VECOUT,
        VECOUT->GM.
    }];

    let arguments = (ins AscendC_AnyTensorType: $dstTensor,
                         Optional<I32>:$dstOffset,
                         AscendC_AnyTensorType: $srcTensor,
                         Optional<I32>:$srcOffset,
                         I32:$tile_length);

    let assemblyFormat =
      " $dstTensor (`[` $dstOffset^ `]`)? `,` $srcTensor (`[` $srcOffset^ `]`)?"
      " `,` $tile_length attr-dict `:` qualified(type($srcTensor)) `to`"
      " qualified(type($dstTensor))";
    let hasVerifier = 1;
}

def DuplicateOp : AscendC_Op<"duplicate"> {
    let summary = "AscendC Duplicate Op";
    let description = [{
        Duplicate a scalar value multiple times and place into the dstTensor
    }];

    let arguments = (ins AscendC_AnyTensorType: $dstTensor,
                         I32: $value,
                         I32: $count);

    let assemblyFormat =
      " $dstTensor `,` $value `,` $count attr-dict `:` qualified(type($dstTensor))";
}

//===----------------------------------------------------------------------===//
// Vector Operation
//===----------------------------------------------------------------------===//

def AscendC_AddOp : AscendC_Op<"add"> {
  let summary = "Addition between AscendC LocalTensors";
  let description = [{
        Perform elementwise addition with LocalTensors:
        dstLocal_i = src0Local_i + srcLocal_i
    }];

  let arguments = (ins AscendC_LocalTensor:$src0,
                       AscendC_LocalTensor:$src1,
                       Optional<I32>:$tile_length,
                       AscendC_LocalTensor:$dst);

  let assemblyFormat = "$dst `,` $src0 `,` $src1 (`,` $tile_length^)?"
    "attr-dict `:` `(` qualified(type($src0)) `,` qualified(type($src1)) `)`"
    " `->` qualified(type($dst))";
}

//===----------------------------------------------------------------------===//
// System Variable Access
//===----------------------------------------------------------------------===//

def GetBlockIdxOp : AscendC_Op<"get_block_idx"> {
    let summary = "AscendC Get Block Idx Op";
    let description = [{
        Get the index for the current core, used with multi-core manipulation
        and calculating multi-core offset.
    }];

    let results = (outs SignlessIntegerLike: $block_id);

    let assemblyFormat = "attr-dict `:` type($block_id)";
}

#endif // MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDC
