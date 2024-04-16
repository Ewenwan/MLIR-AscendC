// RUN: ascendc-mlir-opt  %s | ascendc-mlir-opt  | FileCheck %s

// CHECK-LABEL: func @getBlockIdx
func.func @getBlockIdx() {
    %block_idx = ascendc.get_block_idx : i32
    return
}

// CHECK-LABEL: func @createPipe
func.func @createPipe() {
    %pipe = ascendc.create_pipe : !ascendc.TPipe
    return
}

// CHECK-LABEL: func @createQueue
func.func @createQueue(%pipe: !ascendc.TPipe, %tile_length: i32) {
    %inQueueX = ascendc.create_queue %pipe, %tile_length: !ascendc.TPipe -> !ascendc.TQue<VECIN, 2>
    return
}

// CHECK-LABEL: func @createBuf
func.func @createBuf(%pipe: !ascendc.TPipe, %tile_length: i32) {
    %inQueueX = ascendc.create_tbuf %pipe, %tile_length: !ascendc.TPipe -> !ascendc.TBuf<VECIN>
    return
}

// CHECK-LABEL: func @createGlobalTensor
func.func @createGlobalTensor(%z: !ascendc.GM_ADDR, %offset: i32, %block_len: i32) {
    %zGm = ascendc.create_global_tensor %z, %block_len, %offset: !ascendc.GM_ADDR -> !ascendc.GlobalTensor<?xf32>
    return
}

// CHECK-LABEL: func @dataCopy_gm2ub
func.func @dataCopy_gm2ub(%x: !ascendc.GM_ADDR, %offset: i32, %block_len: i32, %tile_length: i32, %gm_offset: i32) {
    %xGm = ascendc.create_global_tensor %x, %block_len, %offset : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf32>
    %pipe = ascendc.create_pipe : !ascendc.TPipe
    %inQueueX = ascendc.create_queue %pipe, %tile_length : !ascendc.TPipe -> !ascendc.TQue<VECIN, 2>
    %xLocal = ascendc.alloc_tensor(%inQueueX) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf32>
    ascendc.data_copy %xLocal, %xGm[%gm_offset], %tile_length: !ascendc.GlobalTensor<16xf32> to !ascendc.LocalTensor<16xf32>
    return
}

// CHECK-LABEL: func @dataCopy_ub2gm
func.func @dataCopy_ub2gm(%xGm: !ascendc.GlobalTensor<16xf32>, %inQueueX: !ascendc.TQue<VECOUT, 2>, %tile_length: i32, %gm_offset: i32) {
    %xLocal = ascendc.deque(%inQueueX) : !ascendc.TQue<VECOUT, 2> -> !ascendc.LocalTensor<16xf32>
    ascendc.data_copy %xGm[%gm_offset], %xLocal, %tile_length: !ascendc.LocalTensor<16xf32> to !ascendc.GlobalTensor<16xf32>
    return
}

// CHECK-LABEL: func @allocTensor
func.func @allocTensor(%outQueueZ: !ascendc.TQue<VECIN, 2>) {
    // CHECK: %[[zLocal:.*]] = ascendc.alloc_tensor(%[[outQueueZ:.*]]) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<?xf32>
    %zLocal = ascendc.alloc_tensor(%outQueueZ) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<?xf32>
    return
}

// CHECK-LABEL: func @get
func.func @get(%outQueueZ: !ascendc.TBuf<VECIN>) {
    %zLocal = ascendc.get(%outQueueZ) : !ascendc.TBuf<VECIN> -> !ascendc.LocalTensor<?xf32>
    return
}

// CHECK-LABEL: func @allocTensor2
func.func @allocTensor2(%inQueueX: !ascendc.TQue<VECIN, 2>, %alloc_size: i32) {
    // CHECK: %[[xLocal:.*]] = ascendc.alloc_tensor(%[[inQueueX:.*]], %[[alloc_size:.*]]) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<?xf32>
    %xLocal = ascendc.alloc_tensor(%inQueueX, %alloc_size) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<?xf32>
    return
}

// CHECK-LABEL: func @duplicate
func.func @duplicate(%xLocal: !ascendc.LocalTensor<16xf32>, %scalar: i32, %count: i32) {
    ascendc.duplicate %xLocal, %scalar, %count : !ascendc.LocalTensor<16xf32>
    return
}

// CHECK-LABEL: func @freeTensor
func.func @freeTensor(%inQueueX: !ascendc.TQue<VECIN, 2>, %xLocal: !ascendc.LocalTensor<16xf32>) {
    ascendc.free_tensor (%inQueueX, %xLocal) : !ascendc.LocalTensor<16xf32> from !ascendc.TQue<VECIN, 2>
    return
}

// CHECK-LABEL: func @enque
func.func @enque(%zLocal: !ascendc.LocalTensor<16xf32>, %outQueueZ: !ascendc.TQue<VECIN, 2>) {
    ascendc.enque %zLocal, %outQueueZ : !ascendc.LocalTensor<16xf32> to !ascendc.TQue<VECIN, 2>
    return
}

// CHECK-LABEL: func @deque
func.func @deque(%inQueueX: !ascendc.TQue<VECIN, 2>) {
    // CHECK: %[[xLocal:.*]] = ascendc.deque(%[[inQueueX:.*]]) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<?xf32>
    %xLocal = ascendc.deque(%inQueueX) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<?xf32>
    return
}

// CHECK-LABEL: func @add
func.func @add(%xLocal: !ascendc.LocalTensor<16xf32>, %yLocal: !ascendc.LocalTensor<16xf32>, %zLocal: !ascendc.LocalTensor<16xf32>, %tile_length: i32) {
    ascendc.add %zLocal, %xLocal, %yLocal, %tile_length : (!ascendc.LocalTensor<16xf32>, !ascendc.LocalTensor<16xf32>) -> !ascendc.LocalTensor<16xf32>
    return
}

// CHECK-LABEL: func @add2
func.func @add2(%xLocal: !ascendc.LocalTensor<16xf32>, %yLocal: !ascendc.LocalTensor<16xf32>, %zLocal: !ascendc.LocalTensor<16xf32>) {
    ascendc.add %zLocal, %xLocal, %yLocal: (!ascendc.LocalTensor<16xf32>, !ascendc.LocalTensor<16xf32>) -> !ascendc.LocalTensor<16xf32>
    return
}
