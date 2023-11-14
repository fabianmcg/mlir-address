//===- ApplyBarePtr.cpp - Implementation of apply bare pointer -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the bare pointer convention transformation pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Address/IR/AddressDialect.h"
#include "mlir/Dialect/Address/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace addr {
#define GEN_PASS_DEF_ADDRBAREPTRCONVENTION
#include "mlir/Dialect/Address/Transforms/Passes.h.inc"
} // namespace addr
} // namespace mlir

using namespace mlir;
using namespace mlir::addr;

namespace {
struct AddrBarePtrConvention
    : public ::mlir::addr::impl::AddrBarePtrConventionBase<
          AddrBarePtrConvention> {
  using Base::Base;

  void runOnOperation() override;
};

struct FuncInterfacePattern
    : public OpInterfaceRewritePattern<FunctionOpInterface> {
  using Base = OpInterfaceRewritePattern<FunctionOpInterface>;
  using Base::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(FunctionOpInterface op,
                                PatternRewriter &rewriter) const final;
};

struct CallInterfacePattern
    : public OpInterfaceRewritePattern<CallOpInterface> {
  using Base = OpInterfaceRewritePattern<CallOpInterface>;
  using Base::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(CallOpInterface op,
                                PatternRewriter &rewriter) const final;
};

struct ReturnLikePattern : public OpTraitRewritePattern<OpTrait::ReturnLike> {
  using Base = OpTraitRewritePattern<OpTrait::ReturnLike>;
  using Base::OpTraitRewritePattern;
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
FuncInterfacePattern::matchAndRewrite(FunctionOpInterface op,
                                      PatternRewriter &rewriter) const {
  SmallVector<Type> inputTy(op.getArgumentTypes());
  SmallVector<Type> resultTy(op.getResultTypes());
  SmallVector<std::pair<ToMemRefOp, unsigned>> blockArgs;
  // Create placeholder Ops for the block arg update
  Block *entryBlock = &op.getBlocks().front();
  rewriter.setInsertionPoint(entryBlock, entryBlock->begin());
  for (auto arg : op.getArguments()) {
    if (auto mTy = dyn_cast<MemRefType>(arg.getType());
        mTy && mTy.hasStaticShape()) {
      auto null =
          rewriter.create<ConstantOp>(arg.getLoc(), 0, mTy.getMemorySpace());
      auto memref = rewriter.create<ToMemRefOp>(arg.getLoc(), mTy, null);
      blockArgs.push_back({memref, arg.getArgNumber()});
      rewriter.replaceAllUsesWith(arg, memref);
    }
  }
  bool updateResults = false;
  // Update the result types
  for (Type &type : resultTy) {
    if (auto mTy = dyn_cast<MemRefType>(type); mTy && mTy.hasStaticShape()) {
      type = AddressType::get(mTy);
      updateResults = true;
    }
  }
  // Fail if there's nothing to update
  if (!updateResults && blockArgs.empty())
    return failure();

  // Update the function type and block arguments
  rewriter.updateRootInPlace(op, [&]() {
    for (std::pair<ToMemRefOp, unsigned> &arg : blockArgs) {
      BlockArgument blockArg = op.getArgument(arg.second);
      auto nullVal = arg.first.getAddress();
      blockArg.setType(nullVal.getType());
      arg.first.getAddressMutable().set(blockArg);
      rewriter.eraseOp(nullVal.getDefiningOp());
      inputTy[arg.second] = blockArg.getType();
    }
    op.setFunctionTypeAttr(TypeAttr::get(op.cloneTypeWith(inputTy, resultTy)));
  });
  return success();
}

LogicalResult
CallInterfacePattern::matchAndRewrite(CallOpInterface op,
                                      PatternRewriter &rewriter) const {
  // Create place holder Ops for the results
  rewriter.setInsertionPointAfter(op);
  SmallVector<std::pair<ToMemRefOp, unsigned>> results;
  for (OpResult res : op.getOperation()->getResults()) {
    if (auto mTy = dyn_cast<MemRefType>(res.getType());
        mTy && mTy.hasStaticShape()) {
      auto null =
          rewriter.create<ConstantOp>(op.getLoc(), 0, mTy.getMemorySpace());
      auto memref = rewriter.create<ToMemRefOp>(op.getLoc(), mTy, null);
      results.push_back({memref, res.getResultNumber()});
      rewriter.replaceAllUsesWith(res, memref);
    }
  }
  rewriter.setInsertionPoint(op);
  // Create place holder Ops for the arguments
  SmallVector<std::pair<FromMemRefOp, unsigned>> arguments;
  for (OpOperand &operand : op.getArgOperandsMutable()) {
    if (auto mTy = dyn_cast<MemRefType>(operand.get().getType());
        mTy && mTy.hasStaticShape()) {
      auto memref = rewriter.create<FromMemRefOp>(
          op.getLoc(), AddressType::get(mTy), operand.get());
      arguments.push_back({memref, operand.getOperandNumber()});
    }
  }
  // Fail if there's nothing to update
  if (arguments.empty() && results.empty())
    return failure();
  // Update the call operation
  rewriter.updateRootInPlace(op, [&]() {
    MutableOperandRange args = op.getArgOperandsMutable();
    for (std::pair<FromMemRefOp, unsigned> &arg : arguments)
      args[arg.second].set(arg.first);
    for (std::pair<ToMemRefOp, unsigned> &res : results) {
      auto nullVal = res.first.getAddress();
      OpResult result = op.getOperation()->getResult(res.second);
      result.setType(nullVal.getType());
      res.first.getAddressMutable().set(result);
      rewriter.eraseOp(nullVal.getDefiningOp());
    }
  });
  return success();
}

LogicalResult
ReturnLikePattern::matchAndRewrite(Operation *op,
                                   PatternRewriter &rewriter) const {
  SmallVector<unsigned> operandsIndx;
  for (OpOperand &operand : op->getOpOperands())
    if (auto mTy = dyn_cast<MemRefType>(operand.get().getType());
        mTy && mTy.hasStaticShape())
      operandsIndx.push_back(operand.getOperandNumber());
  if (operandsIndx.empty())
    return failure();
  rewriter.updateRootInPlace(op, [&]() {
    MutableArrayRef<OpOperand> opOperands = op->getOpOperands();
    for (unsigned i : operandsIndx) {
      auto address = rewriter.create<FromMemRefOp>(
          op->getLoc(),
          AddressType::get(dyn_cast<MemRefType>(opOperands[i].get().getType())),
          opOperands[i].get());
      opOperands[i].set(address.getResult());
    }
  });
  return success();
}

void AddrBarePtrConvention::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateBarePtrConvetion(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

void mlir::addr::populateBarePtrConvetion(RewritePatternSet &patterns) {
  patterns.add<CallInterfacePattern, FuncInterfacePattern, ReturnLikePattern>(
      patterns.getContext());
}
