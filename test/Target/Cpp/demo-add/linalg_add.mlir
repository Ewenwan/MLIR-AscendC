// RUN: ascendc-mlir-opt %s -legalize-func-for-ascendc -inline -linalg-inline-scalar-operands -test-transform-dialect-interpreter="transform-file-name=%S/add-template.mlir" -canonicalize -ascendc-bufferization-simplification -expand-realloc -buffer-deallocation -canonicalize -buffer-deallocation-simplification -bufferization-lower-deallocations -cse -canonicalize -ascendc-justification -convert-to-ascendc -canonicalize -ascendc-auto-complete -o %t_add_ascendc.mlir -mlir-print-ir-after-all &> %t_add_ascendc.log

// RUN: ascendc-mlir-translate -ascendc-to-cpp %t_add_ascendc.mlir -o %t_add_ascendc.cpp

#map = affine_map<(d0) -> (d0)>
module attributes {torch.debug_module_name = "_lambda"} {
    ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
    func.func @add(%arg0: tensor<16384xf16>, %arg1: tensor<16384xf16>) -> tensor<16384xf16> {
        %4 = tensor.empty() : tensor<16384xf16>
        %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<16384xf16>, tensor<16384xf16>) outs(%4 : tensor<16384xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
            %9 = arith.addf %in, %in_0: f16
            linalg.yield %9 : f16
        } -> tensor<16384xf16>
        return %5 : tensor<16384xf16>
    }
}
