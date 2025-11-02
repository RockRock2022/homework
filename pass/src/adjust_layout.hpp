#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace frontend {

std::unique_ptr<mlir::Pass> createAdjustLayoutPass();

void registerAdjustLayoutPass();

} // namespace frontend
