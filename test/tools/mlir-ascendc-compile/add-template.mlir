transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // Match func op
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">

    // Bufferize
    %tensor_empty_ops = transform.structured.match ops {["tensor.empty"]} in %func : (!transform.op<"func.func">) -> !transform.op<"tensor.empty">
    %buff = transform.bufferization.empty_tensor_to_alloc_tensor %tensor_empty_ops : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
    // Apply one shot bufferize
    %bufferized = transform.bufferization.one_shot_bufferize %func : (!transform.op<"func.func">) -> !transform.any_op

    // Promotion to VECIN
    %bufferized_generics = transform.structured.match ops {["linalg.generic"]} in %bufferized : (!transform.any_op) -> !transform.any_op
    transform.ascendc.promote %bufferized_generics {mapping = [#ascendc.TPosition<VECIN>]} : (!transform.any_op) -> !transform.any_op
    transform.yield
}
