#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "BINOP.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

class BackEnd {
public:
  explicit BackEnd(std::ostream &out);
  int emitMain();
  void init();
  void generate();
  void print(mlir::Value value);
  void printVec(mlir::Value value);

  mlir::Value generateInteger(int value);
  mlir::Value generateValuePtr(mlir::Value value);
  mlir::Value generateRange(mlir::Value lower, mlir::Value upper);
  mlir::Value generateVectorOfSize(mlir::Value size);
  mlir::Value generateVectorFromRange(mlir::Value lower, mlir::Value upper);
  mlir::Value generateVectorToFit(mlir::Value left, mlir::Value right);
  mlir::Value generateVectorToVectorBINOP(mlir::Value left, mlir::Value right,
                                          BINOP op);
  mlir::Value generateVectorToIntegerBINOP(mlir::Value left, mlir::Value right,
                                           BINOP op);
  mlir::Value generateIntegerToVectorBINOP(mlir::Value left, mlir::Value right,
                                           BINOP op);

  mlir::Value generateIndexWithInteger(mlir::Value vector, mlir::Value index);
  mlir::Value generateIndexWithVector(mlir::Value indexee, mlir::Value indexor);

  mlir::Value generateLoadValue(mlir::Value addr);

  mlir::Value generateIntegerBinaryOperation(mlir::Value left,
                                             mlir::Value right, BINOP op);

  mlir::Value generateLoadIdentifierPtr(std::string varName);
  mlir::Value generateLoadIdentifier(std::string varName);

  void generateDeclaration(std::string varName, mlir::Value value);
  void generateAssignment(std::string varName, mlir::Value value);
  void generateInitializeGlobalVar(std::string varName, mlir::Value value);
  void deallocateVectors();

  // LOOP METHOD 2: we either discard method 1 later
  void generateCompAndJump(mlir::Block *trueBlock, mlir::Block *falseBlock,
                           mlir::Value addr);
  mlir::Value
  generateStoreConstant(mlir::Value value); // ret address of the stored value
  void setBuilderInsertionPoint(
      mlir::Block *block); // set insertion point for non shared ptr
  void generateEnterBlock(
      mlir::Block *block); // set insertion point for non shared Ptr
  mlir::Block *generateLoopBegin();
  mlir::Block *generateLoopMiddle(mlir::Value addr);
  mlir::Block *generateBlock();

  // generator helper functions
  mlir::Value getVectorSize(mlir::Value vectorAddr);
  mlir::Value generateGeneratorBegin(mlir::Value domainVecAddr,
                                     std::string domainVar);
  // can be combined with CompAndJump function if we have comparison type in
  // parameter
  mlir::Value generateGeneratorCmpAndJump(mlir::Value domainVecAddr,
                                          mlir::Block *exitBlock,
                                          mlir::Block *trueBlock,
                                          mlir::Value indexAddr,
                                          mlir::Value vectorSize);
  void generateStoreValueInVector(mlir::Value vectorAddr, mlir::Value indexAddr,
                                  mlir::Value exprResult);
  void generateIncrementIndex(
      mlir::Value indexAddr); //  helper function for readability
  void generateSetVectorSize(mlir::Value vecAddr, mlir::Value size);
  void generateUpdateDomainVar(mlir::Value domainVecAddr, mlir::Value indexAddr,
                               std::string domainVar);
  // --------------------------------------------------
  mlir::LLVM::LLVMFuncOp mainFunc;
  mlir::Block *mainEntry;

protected:
  void setupPrint();
  void setupPrintVec();
  void setupVectorRuntime();
  int writeLLVMIR();

private:
  std::vector<std::string> vectorLabels;
  unsigned int allocatedVectors = 0;
  mlir::MLIRContext context;
  mlir::ModuleOp module;
  std::shared_ptr<mlir::OpBuilder> builder;
  std::ostream &out;
  mlir::Location loc;
};