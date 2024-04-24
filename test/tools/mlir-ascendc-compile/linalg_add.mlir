// RUN: mlir-ascendc-compile %s -transform-file %S/add-template.mlir -o %t.cpp

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
