#include "adjust_layout.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
// Generate pass declarations
#define GEN_PASS_DECL
#include "adjust_layout.hpp.inc"
#undef GEN_PASS_DECL

// Generate pass definitions
#define GEN_PASS_DEF_ADJUSTLAYOUTPASS
#include "adjust_layout.hpp.inc"
#undef GEN_PASS_DEF_ADJUSTLAYOUTPASS
} // namespace mlir

// Generate pass registration
#define GEN_PASS_REGISTRATION
#include "adjust_layout.hpp.inc"
#undef GEN_PASS_REGISTRATION

using namespace frontend;

namespace {

bool is4DShape(mlir::Value v) {
  if (auto type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType())) {
    return type.getRank() == 4;
  }
  return false;
}

mlir::Value createTransposeOp(mlir::PatternRewriter &rewriter,
                              mlir::Location loc,
                              llvm::ArrayRef<int32_t> permutation,
                              mlir::Value value) {
  auto valueTy = mlir::cast<mlir::ShapedType>(value.getType());
  auto valueShape = valueTy.getShape();

  auto permType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(permutation.size())}, rewriter.getI32Type());
  auto permAttr = rewriter.getI32TensorAttr(permutation);
  auto permValue =
      rewriter.create<mlir::arith::ConstantOp>(loc, permType, permAttr);

  llvm::SmallVector<int64_t> newShape;
  for (size_t i = 0; i < permutation.size(); ++i) {
    newShape.push_back(valueShape[permutation[i]]);
  }

  auto newValueTy =
      mlir::RankedTensorType::get(newShape, valueTy.getElementType());

  auto transposedOp = rewriter.create<mlir::tosa::TransposeOp>(
      loc, newValueTy, value, permValue);

  return transposedOp.getResult();
}

class AdjustLayoutRewriter
    : public mlir::OpRewritePattern<mlir::tosa::Conv2DOp> {
public:
  explicit AdjustLayoutRewriter(mlir::MLIRContext *ctx)
      : OpRewritePattern(ctx) {}

private:
  bool needsConversion(mlir::tosa::Conv2DOp op) const {
    auto weightType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getWeight().getType());
    if (!weightType || weightType.getRank() != 4)
      return false;

    auto weightShape = weightType.getShape();

    // Simple heuristic: only convert if weight shape is [320, 320, 1, 1]
    // TODO: Check affine_map is a proper NCHW layout
    return weightShape[0] == 320 && weightShape[1] == 320 &&
           weightShape[2] == 1 && weightShape[3] == 1;
  }

public:
  mlir::LogicalResult
  matchAndRewrite(mlir::tosa::Conv2DOp op,
                  mlir::PatternRewriter &rewriter) const override {

    if (!needsConversion(op)) {
      return mlir::failure();
    }

    llvm::outs() << "=== Starting Conv2D layout adjustment ===\n";
    llvm::outs() << "Original op: " << op << "\n";

    mlir::Value newInput = op.getInput();
    if (is4DShape(newInput)) {
      llvm::outs() << "Creating input transpose...\n";
      llvm::SmallVector<int32_t> activationPermutation = {0, 2, 3, 1};
      newInput = createTransposeOp(rewriter, op->getLoc(),
                                   activationPermutation, newInput);
      llvm::outs() << "New input: " << newInput << "\n";
    }

    mlir::Value newWeight = op.getWeight();
    if (is4DShape(newWeight)) {
      llvm::outs() << "Creating weight transpose...\n";
      llvm::SmallVector<int32_t> weightPermutation = {0, 2, 3, 1};
      newWeight = createTransposeOp(rewriter, op->getLoc(), weightPermutation,
                                    newWeight);
      llvm::outs() << "New weight: " << newWeight << "\n";
    }

    // Propgagate output type with a more general way
    // Create a new Conv2D op with NHWC layout for showcase
    llvm::outs() << "Creating new Conv2D op...\n";
    // Calculate new output type [N, H, W, C]
    auto oldOutputType = mlir::cast<mlir::ShapedType>(op.getResult().getType());
    auto oldOutputShape = oldOutputType.getShape();
    llvm::SmallVector<int64_t> newOutputShape = {
        oldOutputShape[0], // N
        oldOutputShape[2], // H
        oldOutputShape[3], // W
        oldOutputShape[1]  // C
    };
    auto newOutputType = mlir::RankedTensorType::get(
        newOutputShape, oldOutputType.getElementType());

    auto padAttr = rewriter.getDenseI64ArrayAttr(op.getPad());
    auto strideAttr = rewriter.getDenseI64ArrayAttr(op.getStride());
    auto dilationAttr = rewriter.getDenseI64ArrayAttr(op.getDilation());
    auto accTypeAttr = mlir::TypeAttr::get(op.getAccType());

    // Create new Conv2D op
    auto newConv2D = rewriter.create<mlir::tosa::Conv2DOp>(
        op->getLoc(), newOutputType, newInput, newWeight, op.getBias(), padAttr,
        strideAttr, dilationAttr, accTypeAttr);

    llvm::outs() << "New Conv2D op: " << newConv2D << "\n";

    llvm::outs() << "Creating output transpose...\n";
    mlir::Value convOutput = newConv2D.getResult();
    llvm::SmallVector<int32_t> outputPermutation = {0, 3, 1, 2}; // NHWC -> NCHW
    convOutput = createTransposeOp(rewriter, op->getLoc(),
                                                outputPermutation, convOutput);
    llvm::outs() << "Final output: " << convOutput << "\n";

    llvm::outs() << "Replacing original op...\n";
    rewriter.replaceOp(op, convOutput);

    llvm::outs() << "=== Layout adjustment completed ===\n";
    return mlir::success();
  }
};

//
// AdjustLayoutPass
//
class AdjustLayoutPass
    : public mlir::impl::AdjustLayoutPassBase<AdjustLayoutPass> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AdjustLayoutRewriter>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//
// Register
//
void frontend::registerAdjustLayoutPass() { ::registerAdjustLayoutPass(); }

//
// createAdjustLayoutPass
//
std::unique_ptr<mlir::Pass> frontend::createAdjustLayoutPass() {
  return std::make_unique<AdjustLayoutPass>();
}
