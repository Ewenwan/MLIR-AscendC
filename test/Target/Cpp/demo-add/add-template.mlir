// transform.sequence failures(propagate) {
// ^bb0(%root: !transform.any_op):
// %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
//   // STEP 1: TILE
//   // Get all of the generics within the function.
//   %generics = transform.structured.match ops {["linalg.generic"]} in %func : (!transform.op<"func.func">) -> !transform.any_op
//   %sub = transform.split_handle %generics : (!transform.any_op) -> (!transform.any_op)
//   // Perform tiling on the add generic, fusing the producers into the tile.
//   %transformed, %loops = transform.structured.fuse %sub {tile_sizes=[1024]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//   // Apply canonicalization to remove redundant code from tiling
//   transform.apply_patterns to %func {
//     transform.apply_patterns.linalg.tiling_canonicalization
//   } : !transform.op<"func.func">

//   // STEP 2: BUFFERIZE
//   // Transform all of the tensor.empty ops into alloc_tensor ops.
//   %tensor_empty_ops = transform.structured.match ops {["tensor.empty"]} in %func : (!transform.op<"func.func">) -> !transform.op<"tensor.empty">
//   %buff = transform.bufferization.empty_tensor_to_alloc_tensor %tensor_empty_ops : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
//   // Apply one shot bufferize
//   %bufferized = transform.bufferization.one_shot_bufferize %func {create_deallocs=false} : (!transform.op<"func.func">) -> !transform.any_op

//   // STEP 3: PROMOTE
//   // Promote all of the buffers in the tiled loop to the gpu memory space
//   %tile_loop = transform.structured.match ops {["scf.for"]} in %bufferized : (!transform.any_op) -> !transform.any_op
//   %bufferized_generics = transform.structured.match ops {["linalg.generic"]} in %tile_loop : (!transform.any_op) -> !transform.any_op
//   transform.ascendc.promote %bufferized_generics {mapping = [#ascendc.TPosition<VECIN>]} : (!transform.any_op) -> !transform.any_op
//   transform.yield
// }




// TRUE NO TILE
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
%func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">

  // STEP 2: BUFFERIZE
  // Transform all of the tensor.empty ops into alloc_tensor ops.
  %tensor_empty_ops = transform.structured.match ops {["tensor.empty"]} in %func : (!transform.op<"func.func">) -> !transform.op<"tensor.empty">
  %buff = transform.bufferization.empty_tensor_to_alloc_tensor %tensor_empty_ops : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
  // Apply one shot bufferize
  %bufferized = transform.bufferization.one_shot_bufferize %func : (!transform.op<"func.func">) -> !transform.any_op

  // STEP 3: PROMOTE
  // Promote all of the buffers in the tiled loop to the gpu memory space
  // %tile_loop = transform.structured.match ops {["scf.for"]} in %bufferized : (!transform.any_op) -> !transform.any_op
  %bufferized_generics = transform.structured.match ops {["linalg.generic"]} in %bufferized : (!transform.any_op) -> !transform.any_op
  transform.ascendc.promote %bufferized_generics {mapping = [#ascendc.TPosition<VECIN>]} : (!transform.any_op) -> !transform.any_op
  transform.yield
}







// // tile whole thing -> remove canolization (no tile?)
// transform.sequence failures(propagate) {
// ^bb0(%func: !transform.op<"func.func">):
//   // STEP 1: TILE
//   // Get all of the generics within the function.
//   %generics = transform.structured.match ops {["linalg.generic"]} in %func : (!transform.op<"func.func">) -> !transform.any_op
//   %sub = transform.split_handle %generics : (!transform.any_op) -> (!transform.any_op)
//   // Perform tiling on the add generic, fusing the producers into the tile.
//   %transformed, %loops = transform.structured.fuse %sub {tile_sizes=[16384]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

//   // STEP 2: BUFFERIZE
//   // Transform all of the tensor.empty ops into alloc_tensor ops.
//   %tensor_empty_ops = transform.structured.match ops {["tensor.empty"]} in %func : (!transform.op<"func.func">) -> !transform.op<"tensor.empty">
//   %buff = transform.bufferization.empty_tensor_to_alloc_tensor %tensor_empty_ops : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
//   // Apply one shot bufferize
//   %bufferized = transform.bufferization.one_shot_bufferize %func {create_deallocs=false} : (!transform.op<"func.func">) -> !transform.any_op

//   // STEP 3: PROMOTE
//   // Promote all of the buffers in the tiled loop to the gpu memory space
//   %tile_loop = transform.structured.match ops {["scf.for"]} in %bufferized : (!transform.any_op) -> !transform.any_op
//   %bufferized_generics = transform.structured.match ops {["linalg.generic"]} in %tile_loop : (!transform.any_op) -> !transform.any_op
//   transform.ascendc.promote %bufferized_generics {mapping = [#ascendc.TPosition<VECIN>]} : (!transform.any_op) -> !transform.any_op
//   transform.yield
// }