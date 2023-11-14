//===- AddrToLLVM.cpp - Implementation of Address to LLVM conversion ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Address to LLVM conversion pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Address/IR/AddressDialect.h"
#include "mlir/Dialect/Address/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace addr {
#define GEN_PASS_DEF_ADDRTOLLVM
#include "mlir/Dialect/Address/Transforms/Passes.h.inc"
} // namespace addr
} // namespace mlir

using namespace mlir;
using namespace mlir::addr;

namespace {
struct AddrToLLVM : public ::mlir::addr::impl::AddrToLLVMBase<AddrToLLVM> {
  using Base::Base;

  void runOnOperation() override;
};

struct AddrTypeConverter : public ::mlir::LLVMTypeConverter {
  AddrTypeConverter(MLIRContext *ctx, const LowerToLLVMOptions &options,
                    const DataLayoutAnalysis *analysis = nullptr)
      : LLVMTypeConverter(ctx, options, analysis) {
    addConversion([&](AddressType type) {
      unsigned as = 0;
      if (auto attr = dyn_cast_or_null<IntegerAttr>(type.getAddressSpace()))
        as = attr.getUInt();
      return LLVM::LLVMPointerType::get(&getContext(), as);
    });
  }
};

struct ConstantOpConversion : public ConvertOpToLLVMPattern<ConstantOp> {
protected:
  using ConvertOpToLLVMPattern<ConstantOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const final;
};

struct TypeOffsetOpConversion : public ConvertOpToLLVMPattern<TypeOffsetOp> {
protected:
  using ConvertOpToLLVMPattern<TypeOffsetOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TypeOffsetOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const final;
};

struct CastOpConversion : public ConvertOpToLLVMPattern<CastOp> {
protected:
  using ConvertOpToLLVMPattern<CastOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const final;
};

struct CastIntOpConversion : public ConvertOpToLLVMPattern<CastIntOp> {
protected:
  using ConvertOpToLLVMPattern<CastIntOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(CastIntOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const final;
};

struct FromMemRefOpConversion : public ConvertOpToLLVMPattern<FromMemRefOp> {
protected:
  using ConvertOpToLLVMPattern<FromMemRefOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(FromMemRefOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const final;
};

struct ToMemRefOpConversion : public ConvertOpToLLVMPattern<ToMemRefOp> {
protected:
  using ConvertOpToLLVMPattern<ToMemRefOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ToMemRefOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const final;
};

struct PtrAddOpConversion : public ConvertOpToLLVMPattern<PtrAddOp> {
protected:
  using ConvertOpToLLVMPattern<PtrAddOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(PtrAddOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult ConstantOpConversion::matchAndRewrite(
    ConstantOp op, OpAdaptor operands,
    ConversionPatternRewriter &rewriter) const {
  auto cst = rewriter.create<LLVM::ConstantOp>(
      op.getLoc(), rewriter.getIntegerAttr(
                       typeConverter->convertType(op.getValueAttr().getType()),
                       operands.getValue()));
  // Convert the constant to a ptr
  rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(
      op, typeConverter->convertType(op.getType()), cst);
  return success();
}

LogicalResult TypeOffsetOpConversion::matchAndRewrite(
    TypeOffsetOp op, OpAdaptor operands,
    ConversionPatternRewriter &rewriter) const {
  // Use GEP to compute the type offset
  const LLVMTypeConverter *tc =
      static_cast<const LLVMTypeConverter *>(typeConverter);
  auto ptrTy = LLVM::LLVMPointerType::get(getContext());
  Value nullOp = rewriter.create<LLVM::ZeroOp>(op.getLoc(), ptrTy);
  auto offset = rewriter.create<LLVM::GEPOp>(
      op.getLoc(), ptrTy, tc->convertType(op.getBaseType()), nullOp,
      ArrayRef<LLVM::GEPArg>({LLVM::GEPArg(1)}));
  rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(
      op, tc->convertType(op.getType()), offset.getRes());
  return success();
}

LogicalResult
CastOpConversion::matchAndRewrite(CastOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(
      op, typeConverter->convertType(op.getType()), operands.getInput());
  return success();
}

LogicalResult CastIntOpConversion::matchAndRewrite(
    CastIntOp op, OpAdaptor operands,
    ConversionPatternRewriter &rewriter) const {
  if (op.getType().isIntOrIndex())
    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(
        op, typeConverter->convertType(op.getType()), operands.getInput());
  else
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(
        op, typeConverter->convertType(op.getType()), operands.getInput());
  return success();
}

LogicalResult FromMemRefOpConversion::matchAndRewrite(
    FromMemRefOp op, OpAdaptor operands,
    ConversionPatternRewriter &rewriter) const {
  MemRefDescriptor descriptor(operands.getInput());
  if (op.getExtractBase())
    rewriter.replaceOp(op, descriptor.allocatedPtr(rewriter, op.getLoc()));
  else
    rewriter.replaceOp(op, descriptor.alignedPtr(rewriter, op.getLoc()));
  return success();
}

LogicalResult ToMemRefOpConversion::matchAndRewrite(
    ToMemRefOp op, OpAdaptor operands,
    ConversionPatternRewriter &rewriter) const {
  const LLVMTypeConverter *tc =
      static_cast<const LLVMTypeConverter *>(typeConverter);
  Value descriptor;
  if (operands.getBase())
    descriptor = MemRefDescriptor::fromStaticShape(
        rewriter, op.getLoc(), *tc, op.getType(), operands.getAddress(),
        operands.getBase());
  else
    descriptor = MemRefDescriptor::fromStaticShape(
        rewriter, op.getLoc(), *tc, op.getType(), operands.getAddress());
  rewriter.replaceOp(op, descriptor);
  return success();
}

LogicalResult
PtrAddOpConversion::matchAndRewrite(PtrAddOp op, OpAdaptor operands,
                                    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
      op, operands.getBase().getType(), rewriter.getI8Type(),
      operands.getBase(), operands.getOffset());
  return success();
}

void AddrToLLVM::runOnOperation() {
  ModuleOp module = getOperation();
  StringRef dataLayout;
  auto dataLayoutAttr = dyn_cast_or_null<StringAttr>(
      module->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName()));
  if (dataLayoutAttr)
    dataLayout = dataLayoutAttr.getValue();
  if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
          dataLayout, [this](const Twine &message) {
            getOperation().emitError() << message.str();
          }))) {
    signalPassFailure();
    return;
  }
  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(&getContext(),
                             dataLayoutAnalysis.getAtOrAbove(module));
  AddrTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);
  LLVMConversionTarget target(getContext());
  std::optional<SymbolTable> optSymbolTable = std::nullopt;
  const SymbolTable *symbolTable = nullptr;
  if (!options.useBarePtrCallConv) {
    optSymbolTable.emplace(module);
    symbolTable = &optSymbolTable.value();
  }
  RewritePatternSet patterns(&getContext());
  patterns.insert<ConstantOpConversion, TypeOffsetOpConversion,
                  CastOpConversion, CastIntOpConversion, FromMemRefOpConversion,
                  ToMemRefOpConversion, PtrAddOpConversion>(typeConverter);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns, symbolTable);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
