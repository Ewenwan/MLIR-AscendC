// RUN: ascendc-mlir-translate -ascendc-to-cpp %s | FileCheck %s

// CHECK: add
func.func @add_custom(%x : !ascendc.GM_ADDR, %y : !ascendc.GM_ADDR, %z : !ascendc.GM_ADDR) {
// Struct Constructor
    %total_len = arith.constant 16384 : i32

    %pipe = ascendc.create_pipe : !ascendc.TPipe

    %xGm = ascendc.create_global_tensor %x, %total_len : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf16>
    %yGm = ascendc.create_global_tensor %y, %total_len : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf16>
    %zGm = ascendc.create_global_tensor %z, %total_len : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf16>

    %inQueueX = ascendc.create_queue %pipe, %total_len : !ascendc.TPipe -> !ascendc.TQue<VECIN, 2>
    %inQueueY = ascendc.create_queue %pipe, %total_len : !ascendc.TPipe -> !ascendc.TQue<VECIN, 2>
    %outQueueZ = ascendc.create_queue %pipe, %total_len : !ascendc.TPipe -> !ascendc.TQue<VECOUT, 2>

// Process
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32

    // Copy in
    %xLocal = ascendc.alloc_tensor(%inQueueX) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf16>
    %yLocal = ascendc.alloc_tensor(%inQueueY) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf16>
    ascendc.data_copy %xLocal, %xGm, %total_len : !ascendc.GlobalTensor<16xf16> to !ascendc.LocalTensor<16xf16>
    ascendc.data_copy %yLocal, %yGm, %total_len : !ascendc.GlobalTensor<16xf16> to !ascendc.LocalTensor<16xf16>
    ascendc.enque %xLocal, %inQueueX : !ascendc.LocalTensor<16xf16> to !ascendc.TQue<VECIN, 2>
    ascendc.enque %yLocal, %inQueueY : !ascendc.LocalTensor<16xf16> to !ascendc.TQue<VECIN, 2>

    // Compute
    %xLocal_deque = ascendc.deque(%inQueueX) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf16>
    %yLocal_deque = ascendc.deque(%inQueueY) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf16>
    %zLocal = ascendc.alloc_tensor(%outQueueZ) : !ascendc.TQue<VECOUT, 2> -> !ascendc.LocalTensor<16xf16>
    ascendc.add %zLocal, %xLocal_deque, %yLocal_deque, %total_len : (!ascendc.LocalTensor<16xf16>, !ascendc.LocalTensor<16xf16>) -> !ascendc.LocalTensor<16xf16>
    ascendc.enque %zLocal, %outQueueZ : !ascendc.LocalTensor<16xf16> to !ascendc.TQue<VECOUT, 2>
    ascendc.free_tensor (%inQueueX, %xLocal_deque) : !ascendc.LocalTensor<16xf16> from !ascendc.TQue<VECIN, 2>
    ascendc.free_tensor (%inQueueY, %yLocal_deque) : !ascendc.LocalTensor<16xf16> from !ascendc.TQue<VECIN, 2>

    // Copy out
    %zLocal_deque = ascendc.deque(%outQueueZ) : !ascendc.TQue<VECOUT, 2> -> !ascendc.LocalTensor<16xf16>
    ascendc.data_copy %zGm, %zLocal_deque, %total_len : !ascendc.LocalTensor<16xf16> to !ascendc.GlobalTensor<16xf16>
    ascendc.free_tensor (%outQueueZ, %zLocal_deque) : !ascendc.LocalTensor<16xf16> from !ascendc.TQue<VECOUT, 2>

    return
}
