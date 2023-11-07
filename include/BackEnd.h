#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"
#include "BuiltinTypes/BuiltInTypes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

class BackEnd {
public:
  explicit BackEnd(std::ostream &out);
  int emitMain();
  void init();
  void generate();
  void print(mlir::Value value);
  void printVec(mlir::Value value);
  void printCommonType(mlir::Value value);

  mlir::Value generateInteger(int value);
  mlir::Value generateValue(int value);
  mlir::Value generateValue(float value);
  mlir::Value generateValue(char* value);
  mlir::Value generateValue(char value);
  mlir::Value generateValue(bool value);

  void functionShowcase();

  // construct tuple from values
  mlir::Value generateValue(std::vector<mlir::Value> values);

  mlir::Value performBINOP(mlir::Value left, mlir::Value right, BINOP op);
  mlir::Value performUNARYOP(mlir::Value value, UNARYOP op);


  mlir::Value generateValuePtr(mlir::Value value);
  mlir::Value generateRange(mlir::Value lower, mlir::Value upper);
  mlir::Value generateVectorOfSize(mlir::Value size);
  mlir::Value generateVectorFromRange(mlir::Value lower, mlir::Value upper);
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
  void deallocateObjects();

  // LOOP METHOD 2: we either discard method 1 later
  void generateCompAndJump(mlir::Block *trueBlock, mlir::Block *falseBlock,
                           mlir::Value addr);
  mlir::Value
  generateStoreConstant(mlir::Value value); // ret address of the stored value
  void setBuilderInsertionPoint(
      mlir::Block *block); // set insertion point for non shared ptr
  void generateEnterBlock(
      mlir::Block *block); // set insertion point for non shared Ptr
  bool conditionalJumpToBlock(mlir::Block *block, bool ifJump); // (statically) conditionally jump to a block
  mlir::Block *generateLoopBegin();
  mlir::Block *generateLoopMiddle(mlir::Value addr);
  mlir::Block *generateBlock();

  // Loop and if conditional helper - downcasting to bool
  mlir::Value downcastToBool(mlir::Value val);

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
  void setupCommonTypeRuntime();
  int writeLLVMIR();

private:
  std::vector<std::string> vectorLabels;
  std::vector<std::string> objectLabels;

  unsigned int allocatedVectors = 0;
  unsigned int allocatedObjects = 0;

  mlir::Value generateCommonType(mlir::Value value, int Type);

  mlir::MLIRContext context;
  mlir::ModuleOp module;
  std::shared_ptr<mlir::OpBuilder> builder;
  std::ostream &out;
  mlir::Location loc;
};
