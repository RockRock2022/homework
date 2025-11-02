#ifndef ADJUST_LAYOUT_TD
#define ADJUST_LAYOUT_TD

include "mlir/Pass/PassBase.td"

def AdjustLayoutPass : Pass<"adjust-layout", "mlir::func::FuncOp"> {
  let summary = "Adjust layout pass";

  let description = [{
    This pass adjusts the layout of operations by converting from NCHW to NHWC format
    for better hardware performance and memory locality.
    
    The pass specifically targets TOSA Conv2D operations and inserts transpose
    operations to convert between data formats:
    - Input: NCHW -> NHWC
    - Weight: [OC, IC, H, W] -> [OC, H, W, IC] 
    - Output: NHWC -> NCHW
  }];
  
  let dependentDialects = [
    "mlir::tosa::TosaDialect"
  ];
  
  let constructor = "frontend::createAdjustLayoutPass()";
}

#endif // ADJUST_LAYOUT_TD
