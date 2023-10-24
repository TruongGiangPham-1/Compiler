#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

class BackEnd {
 public:
    BackEnd();

    int emitMain();

 protected:
    void setupPrintf();
    void printNewline();

 private:
    mlir::MLIRContext context;
    mlir::ModuleOp module;
    std::shared_ptr<mlir::OpBuilder> builder;

    mlir::Location loc;
};
