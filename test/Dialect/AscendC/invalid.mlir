// RUN: ascendc-mlir-opt  %s -verify-diagnostics

func.func @Add_Custom(%x : !ascendc.GM_ADDR, %y : !ascendc.GM_ADDR, %z : !ascendc.GM_ADDR) {
// Compute Offset
    %total_len = arith.constant 98304 : i32
    %use_core_num = arith.constant 8 : i32
    %block_len = arith.divui %total_len, %use_core_num : i32
    %tile_num = arith.constant 16 : i32
    %buffer_num = arith.constant 2 : i32
    %tile_length_pre_buffer = arith.divui %block_len, %tile_num : i32
    %tile_length = arith.divui %tile_length_pre_buffer, %buffer_num : i32

    %block_idx = ascendc.get_block_idx : i32
    %offset = arith.muli %block_len, %block_idx : i32

// Struct Setup
    %xGm = ascendc.create_global_tensor %x, %block_len, %offset : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf32>
    %yGm = ascendc.create_global_tensor %y, %block_len, %offset : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf32>
    %zGm = ascendc.create_global_tensor %z, %block_len, %offset : !ascendc.GM_ADDR -> !ascendc.GlobalTensor<16xf32>

    %pipe = ascendc.create_pipe : !ascendc.TPipe
    %inQueueX = ascendc.create_queue %pipe, %tile_length : !ascendc.TPipe -> !ascendc.TQue<VECIN, 2>
    %inQueueY = ascendc.create_queue %pipe, %tile_length : !ascendc.TPipe -> !ascendc.TQue<A2, 2>
    %outQueueZ = ascendc.create_queue %pipe, %tile_length : !ascendc.TPipe -> !ascendc.TQue<VECOUT, 2>

// Process
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %loop_count = arith.muli %tile_num, %buffer_num : i32
    scf.for %progress = %c0 to %loop_count step %c1 : i32 {
        %gm_offset = arith.muli %progress, %tile_length : i32

        // Copy in
        %xLocal = ascendc.alloc_tensor(%inQueueX) : !ascendc.TQue<VECIN, 2> -> !ascendc.LocalTensor<16xf32>
        %yLocal = ascendc.alloc_tensor(%inQueueY) : !ascendc.TQue<A2, 2> -> !ascendc.LocalTensor<16xf32>
        ascendc.data_copy %xLocal, %xGm[%gm_offset], %tile_length : !ascendc.GlobalTensor<16xf32> to !ascendc.LocalTensor<16xf32>
        // expected-error @below{{'ascendc.data_copy' op unsupported data copy path}}
        ascendc.data_copy %yLocal, %yGm[%gm_offset], %tile_length : !ascendc.GlobalTensor<16xf32> to !ascendc.LocalTensor<16xf32>

        // Compute
        %zLocal = ascendc.alloc_tensor(%outQueueZ) : !ascendc.TQue<VECOUT, 2> -> !ascendc.LocalTensor<16xf32>
        ascendc.add %zLocal, %xLocal, %yLocal, %tile_length : (!ascendc.LocalTensor<16xf32>, !ascendc.LocalTensor<16xf32>) -> !ascendc.LocalTensor<16xf32>
        ascendc.add %zLocal, %xLocal, %zLocal, %tile_length : (!ascendc.LocalTensor<16xf32>, !ascendc.LocalTensor<16xf32>) -> !ascendc.LocalTensor<16xf32>

        // Copy out
        ascendc.data_copy %zGm[%gm_offset], %zLocal, %tile_length : !ascendc.LocalTensor<16xf32> to !ascendc.GlobalTensor<16xf32>
        scf.yield
    }
    return
}
