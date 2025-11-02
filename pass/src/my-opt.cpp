#include "adjust_layout.hpp"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::func::FuncDialect>();

  frontend::registerAdjustLayoutPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "my-opt", registry));
}
