#pragma once

#include "Types/TYPES.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ValueRange.h"

class BackEnd {
public:
  explicit BackEnd(std::ostream &out);
  int emitMain();
  void init();
  void generate();
  void print(mlir::Value value);
  void printVec(mlir::Value value);
  void printCommonType(mlir::Value value);
  void streamOut(mlir::Value value);
  void streamIn(mlir::Value value);
  void verifyFunction(int line, std::string name);

  mlir::Value generateInteger(int value);
  mlir::Value generateValue(int value);
  mlir::Value generateValue(float value);
  mlir::Value generateValue(char value);
  mlir::Value generateValue(bool value);
  mlir::Value generateValue(std::string value);
  mlir::Value generateValue(mlir::Value lower, mlir::Value upper);
  mlir::Value generateValue(unsigned length);


  void functionShowcase();

  // construct tuple from values
  mlir::Value generateValue(std::vector<mlir::Value> values);

  mlir::Value performBINOP(mlir::Value left, mlir::Value right, BINOP op);
  mlir::Value performUNARYOP(mlir::Value value, UNARYOP op);
  mlir::Value generateCallNamed(std::string signature, std::vector<mlir::Value> arguments);

  mlir::Value generateValuePtr(mlir::Value value);
  mlir::Value generateRange(mlir::Value lower, mlir::Value upper);
  mlir::Value generateVectorOfSize(mlir::Value size);
  mlir::Value generateVectorFromRange(mlir::Value lower, mlir::Value upper);
  mlir::Value generateIndexWithInteger(mlir::Value vector, mlir::Value index);
  mlir::Value generateIndexWithVector(mlir::Value indexee, mlir::Value indexor);
  mlir::Value copyCommonType(mlir::Value val);


  mlir::Value generateLoadValue(mlir::Value addr);
  mlir::Value generateNullValue(TYPE type);
  mlir::Value generateIdentityValue(TYPE type);

  mlir::Value generateIntegerBinaryOperation(mlir::Value left,
                                             mlir::Value right, BINOP op);

  mlir::Value cast(mlir::Value from, TYPE toType);
  void appendCommon(mlir::Value destination, mlir::Value item);

  mlir::Value possiblyCast(mlir::Value val, std::shared_ptr<Type> nullableType);
  mlir::Block* generateFunctionDefinition(std::string signature, size_t argumentSize, bool isVoid);

  void generateEndFunctionDefinition(mlir::Block* returnBlock, int line);
  void generateReturn(mlir::Value returnVal);

  mlir::Value generateLoadIdentifierPtr(std::string varName);
  mlir::Value generateLoadIdentifier(std::string varName);
  mlir::Value generateLoadArgument(size_t index);
  mlir::Value indexCommonType(mlir::Value indexee, int indexor);

  void generateDeclaration(std::string varName, mlir::Value value);
  void generateAssignment(std::string varName, mlir::Value value);
  void generateAssignment(mlir::Value ptr, mlir::Value value);

  void generateInitializeGlobalVar(std::string varName, mlir::Value value);
  void deallocateVectors();
  void deallocateObjects();
  void pushScope();
  void popScope();



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
  // global scope is functionStack[0]
  // current scope is functionStack.back()
  std::vector<mlir::LLVM::LLVMFuncOp> functionStack;
  mlir::Block *mainEntry;

protected:
  void setupPrint();
  void setupPrintVec();
  void setupVectorRuntime();
  void setupCommonTypeRuntime();
  std::string trackObject();
  mlir::Value translateToMLIRType(TYPE type);
  int writeLLVMIR();

private:
  std::vector<std::string> vectorLabels;
  // stack of labels defined in the current scope
  std::vector<std::vector<std::string>*> objectLabels;
  std::vector<mlir::LLVM::LLVMFuncOp> functionContext;

  unsigned int allocatedVectors = 0;
  unsigned int allocatedObjects = 0;

  mlir::Value generateCommonType(mlir::Value value, int Type);

  mlir::MLIRContext context;
  mlir::ModuleOp module;
  std::shared_ptr<mlir::OpBuilder> builder;
  std::ostream &out;
  mlir::Location loc;
};
