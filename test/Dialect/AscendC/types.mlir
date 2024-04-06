// RUN: ascendc-mlir-opt  %s | ascendc-mlir-opt  | FileCheck %s

module {

    // CHECK: func @ascendc_tpipe(%arg0: !ascendc.TPipe)
    func.func @ascendc_tpipe(%arg0: !ascendc.TPipe) {
        return
    }

    // CHECK: func @ascendc_gm_addr(%arg0: !ascendc.GM_ADDR)
    func.func @ascendc_gm_addr(%arg0: !ascendc.GM_ADDR) {
        return
    }

    // CHECK: func @ascendc_tque(%arg0: !ascendc.TQue<VECIN, 2>)
    func.func @ascendc_tque(%arg0: !ascendc.TQue<VECIN, 2>) {
        return
    }

    // CHECK: func @ascendc_tbuf(%arg0: !ascendc.TBuf<VECIN>)
    func.func @ascendc_tbuf(%arg0: !ascendc.TBuf<VECIN>) {
        return
    }

    // CHECK: func @ascendc_localtensor(%arg0: !ascendc.LocalTensor<16xf32>)
    func.func @ascendc_localtensor(%arg0: !ascendc.LocalTensor<16xf32>) {
        return
    }

    // CHECK: func @ascendc_localtensor2(%arg0: !ascendc.LocalTensor<?x?xf32>)
    func.func @ascendc_localtensor2(%arg0: !ascendc.LocalTensor<?x?xf32>) {
        return
    }

    // CHECK: func @ascendc_localtensor3(%arg0: !ascendc.LocalTensor<*xf32>)
    func.func @ascendc_localtensor3(%arg0: !ascendc.LocalTensor<*xf32>) {
        return
    }

    // CHECK: func @ascendc_globaltensor(%arg0: !ascendc.GlobalTensor<2x2xf32>)
    func.func @ascendc_globaltensor(%arg0: !ascendc.GlobalTensor<2x2xf32>) {
        return
    }
}
