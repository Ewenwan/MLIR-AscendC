// RUN: ascendc-mlir-opt  %s | ascendc-mlir-opt  | FileCheck %s

// CHECK-LABEL: func @Add_Custom
func.func @Add_Custom(%x : !ascendc.GM_ADDR, %y : !ascendc.GM_ADDR, %z : !ascendc.GM_ADDR) {
// Struct Constructor
    %total_len = arith.constant 98304 : i32
    %use_core_num = arith.constant 8 : i32
    %block_len = arith.divui %total_len, %use_core_num : i32
    %tile_num = arith.constant 16 : i32
    %buffer_num = arith.constant 2 : i32
    %tile_length_pre_buffer = arith.divui %block_len, %tile_num : i32
    %tile_length = arith.divui %tile_length_pre_buffer, %buffer_num : i32

    %block_idx = ascendc.get_block_idx : i32
    %offset = arith.muli %block_len, %block_idx : i32
    %xGm = ascendc.create_global_tensor %x, %block_len, %offset : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf32>
    %yGm = ascendc.create_global_tensor %y, %block_len, %offset : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf32>
    %zGm = ascendc.create_global_tensor %z, %block_len, %offset : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf32>

    %pipe = ascendc.create_pipe : !ascendc.TPipe
    %inQueueX = ascendc.create_queue %pipe, %tile_length : !ascendc.TPipe -> !ascendc.TQue<VECIN, 2>
    %inQueueY = ascendc.create_queue %pipe, %tile_length : !ascendc.TPipe -> !ascendc.TQue<VECIN, 2>
    %outQueueZ = ascendc.create_queue %pipe, %tile_length : !ascendc.TPipe -> !ascendc.TQue<VECOUT, 2>

// Process
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %loop_count = arith.muli %tile_num, %buffer_num : i32
    scf.for %progress = %c0 to %loop_count step %c1 : i32 {
        %gm_offset = arith.muli %progress, %tile_length : i32

        // Copy in
        %xLocal = ascendc.alloc_tensor(%inQueueX) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf32>
        %yLocal = ascendc.alloc_tensor(%inQueueY) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf32>
        ascendc.data_copy %xLocal, %xGm[%gm_offset], %tile_length : !ascendc.GlobalTensor<16xf32> to !ascendc.LocalTensor<16xf32>
        ascendc.data_copy %yLocal, %yGm[%gm_offset], %tile_length : !ascendc.GlobalTensor<16xf32> to !ascendc.LocalTensor<16xf32>
        ascendc.enque %xLocal, %inQueueX : !ascendc.LocalTensor<16xf32> to !ascendc.TQue<VECIN, 2>
        ascendc.enque %yLocal, %inQueueY : !ascendc.LocalTensor<16xf32> to !ascendc.TQue<VECIN, 2>

        // Compute
        %xLocal_deque = ascendc.deque(%inQueueX) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf32>
        %yLocal_deque = ascendc.deque(%inQueueY) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf32>
        %zLocal = ascendc.alloc_tensor(%outQueueZ) : !ascendc.TQue<VECOUT, 2> -> !ascendc.LocalTensor<16xf32>
        ascendc.add %zLocal, %xLocal_deque, %yLocal_deque, %tile_length : (!ascendc.LocalTensor<16xf32>, !ascendc.LocalTensor<16xf32>) -> !ascendc.LocalTensor<16xf32>
        ascendc.enque %zLocal, %outQueueZ : !ascendc.LocalTensor<16xf32> to !ascendc.TQue<VECOUT, 2>
        ascendc.free_tensor (%inQueueX, %xLocal_deque) : !ascendc.LocalTensor<16xf32> from !ascendc.TQue<VECIN, 2>
        ascendc.free_tensor (%inQueueY, %yLocal_deque) : !ascendc.LocalTensor<16xf32> from !ascendc.TQue<VECIN, 2>

        // Copy out
        %zLocal_deque = ascendc.deque(%outQueueZ) : !ascendc.TQue<VECOUT, 2> -> !ascendc.LocalTensor<16xf32>
        ascendc.data_copy %zGm[%gm_offset], %zLocal_deque, %tile_length : !ascendc.LocalTensor<16xf32> to !ascendc.GlobalTensor<16xf32>
        ascendc.free_tensor (%outQueueZ, %zLocal_deque) : !ascendc.LocalTensor<16xf32> from !ascendc.TQue<VECOUT, 2>

        scf.yield
    }
    return
}

// func.func @Add_Custom(%arg0: !ascendc.GM_ADDR, %arg1: !ascendc.GM_ADDR, %arg2: !ascendc.GM_ADDR) {
//     %c98304_i32 = arith.constant 98304 : i32
//     %c8_i32 = arith.constant 8 : i32
//     %0 = arith.divui %c98304_i32, %c8_i32 : i32
//     %c16_i32 = arith.constant 16 : i32
//     %c2_i32 = arith.constant 2 : i32
//     %1 = arith.divui %0, %c16_i32 : i32
//     %2 = arith.divui %1, %c2_i32 : i32
//     %3 = ascendc.get_block_idx : i32
//     %4 = arith.muli %0, %3 : i32
//     %5 = ascendc.create_global_tensor %arg0, %4, %0 : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf32>
//     %6 = ascendc.create_global_tensor %arg1, %4, %0 : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf32>
//     %7 = ascendc.create_global_tensor %arg2, %4, %0 : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf32>
//     %8 = ascendc.create_pipe : !ascendc.TPipe
//     %9 = ascendc.create_queue %8, %2 : !ascendc.TPipe -> !ascendc.TQue<VECIN, 2>
//     %10 = ascendc.create_queue %8, %2 : !ascendc.TPipe -> !ascendc.TQue<VECIN, 2>
//     %11 = ascendc.create_queue %8, %2 : !ascendc.TPipe -> !ascendc.TQue<VECOUT, 2>
//     %c0_i32 = arith.constant 0 : i32
//     %c1_i32 = arith.constant 1 : i32
//     %12 = arith.muli %c16_i32, %c2_i32 : i32
//     scf.for %arg3 = %c0_i32 to %12 step %c1_i32  : i32 {
//       %13 = arith.muli %arg3, %2 : i32
//       %14 = ascendc.alloc_tensor(%9) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf32>
//       %15 = ascendc.alloc_tensor(%10) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf32>
//       ascendc.data_copy %14, %5[%13], %2 : !ascendc.GlobalTensor<16xf32> to !ascendc.LocalTensor<16xf32>
//       ascendc.data_copy %15, %6[%13], %2 : !ascendc.GlobalTensor<16xf32> to !ascendc.LocalTensor<16xf32>
//       ascendc.enque %14, %9 : !ascendc.LocalTensor<16xf32> to !ascendc.TQue<VECIN, 2>
//       ascendc.enque %15, %10 : !ascendc.LocalTensor<16xf32> to !ascendc.TQue<VECIN, 2>
//       %16 = ascendc.deque(%9) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf32>
//       %17 = ascendc.deque(%10) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf32>
//       %18 = ascendc.alloc_tensor(%11) : !ascendc.TQue<VECOUT, 2> -> !ascendc.LocalTensor<16xf32>
//       ascendc.add %18, %16, %17, %2 : (!ascendc.LocalTensor<16xf32>, !ascendc.LocalTensor<16xf32>) -> !ascendc.LocalTensor<16xf32>
//       ascendc.enque %18, %11 : !ascendc.LocalTensor<16xf32> to !ascendc.TQue<VECOUT, 2>
//       ascendc.free_tensor(%9, %16) : !ascendc.LocalTensor<16xf32> from !ascendc.TQue<VECIN, 2>
//       ascendc.free_tensor(%10, %17) : !ascendc.LocalTensor<16xf32> from !ascendc.TQue<VECIN, 2>
//       %19 = ascendc.deque(%11) : !ascendc.TQue<VECOUT, 2> -> !ascendc.LocalTensor<16xf32>
//       ascendc.data_copy %7[%13], %19, %2 : !ascendc.LocalTensor<16xf32> to !ascendc.GlobalTensor<16xf32>
//       ascendc.free_tensor(%11, %19) : !ascendc.LocalTensor<16xf32> from !ascendc.TQue<VECOUT, 2>
//     }
//     return
//   }
