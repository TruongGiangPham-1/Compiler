#include "llvm/ADT/APFloat.h"
#include "BuiltinTypes/BuiltInTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include <assert.h>
#include <iostream>
#include <string>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"

// need these to translate LLVM dialect MLIR to LLVM IR
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// need to verify LLVM IR module
#include "llvm/IR/Verifier.h"

// need to output LLVM IR module to ostream
#include "llvm/Support/raw_os_ostream.h"

// need to declare a runtime
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"

#include "Operands/BINOP.h"
#include "BackEnd.h"

#define DEBUG

/**
 *  Set up main function
 */
void BackEnd::init() {
#ifdef DEBUG
  std::cout << "BEGINNING CODEGEN" << std::endl;
#endif

  auto intType = builder->getI32Type();
  auto mainType = mlir::LLVM::LLVMFunctionType::get(intType, {}, false);

  mainFunc = builder->create<mlir::LLVM::LLVMFuncOp>(loc, "main", mainType);

  mainEntry = mainFunc.addEntryBlock();
  builder->setInsertionPointToStart(mainEntry);
}

/**
 * Finish codegen + main function.
 */
void BackEnd::generate() {
#ifdef DEBUG
  std::cout << "CODEGEN FINISHED, ending main function and outputting"
            << std::endl;
#endif


  auto intType = builder->getI32Type();

  mlir::Value zero = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getIntegerAttr(intType, 0));
  builder->create<mlir::LLVM::ReturnOp>(loc, zero);

  if (mlir::failed(mlir::verify(
          module))) { // trying to verify will complain about some issue that
                      // did not exist when I dump it in visitLoop()
    module.emitError("module failed to verify");
  }
  int result = this->writeLLVMIR();
  if (result != 0) {
    std::cerr << "Failed to write LLVM IR" << std::endl;
  }
}

int BackEnd::writeLLVMIR() {
  mlir::registerLLVMDialectTranslation(context);

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    std::cerr << "Failed to translate to LLVM IR" << std::endl;
    return -1;
  }

  llvm::verifyModule(*llvmModule, &llvm::errs());

  std::cout << "==============================================================="
               "=================\n";
  std::cout << "LLVM IR\n";
  std::cout << "-------\n";
  llvmModule->dump();
  std::cout << "==============================================================="
               "=================\n";

  // print LLVM IR to file
  llvm::raw_os_ostream output(this->out);
  output << *llvmModule;

  return 0;
}

BackEnd::BackEnd(std::ostream &out)
    : out(out), loc(mlir::UnknownLoc::get(&context)) {
  context.loadDialect<mlir::LLVM::LLVMDialect>();

  builder = std::make_shared<mlir::OpBuilder>(&context);

  // Open a new context and module.
  module = mlir::ModuleOp::create(builder->getUnknownLoc());
  builder->setInsertionPointToStart(module.getBody());

  setupCommonTypeRuntime();
}

int BackEnd::emitMain() {
  mlir::Type intType = mlir::IntegerType::get(&context, 32);
  auto mainType = mlir::LLVM::LLVMFunctionType::get(intType, {}, false);
  auto mainFunc =
      builder->create<mlir::LLVM::LLVMFuncOp>(loc, "main", mainType);
  mlir::Block *entry = mainFunc.addEntryBlock();
  builder->setInsertionPointToStart(entry);

  mlir::Value zero = builder->create<mlir::LLVM::ConstantOp>(
      loc, intType, builder->getIntegerAttr(intType, 0));
  builder->create<mlir::LLVM::ReturnOp>(builder->getUnknownLoc(), zero);


  if (mlir::failed(mlir::verify(module))) {
    module.emitError("module failed to verify");
    return -1;
  }
  return 0;
}

void BackEnd::setupCommonTypeRuntime() {
  auto voidType = mlir::LLVM::LLVMVoidType::get(&context);
  auto boolType = builder->getI1Type();
  auto intType = builder->getI32Type();

  auto intPtrType = mlir::LLVM::LLVMPointerType::get(intType);

  // mlir doesn't allow void types. we do a little hacking
  auto voidPtrType = mlir::LLVM::LLVMPointerType::get(intType);

  auto commonType =
      mlir::LLVM::LLVMStructType::getLiteral(&context, {intType, intPtrType});

  auto commonTypeAddr = mlir::LLVM::LLVMPointerType::get(commonType);
  
  auto tupleType =
      mlir::LLVM::LLVMStructType::getLiteral(&context, {intType, intType, mlir::LLVM::LLVMPointerType::get(commonTypeAddr)});
  auto tupleTypeAddr = mlir::LLVM::LLVMPointerType::get(tupleType);


  auto printType = mlir::LLVM::LLVMFunctionType::get(
      voidType, {commonTypeAddr});

  auto promoteCommonTypeFuncType = mlir::LLVM::LLVMFunctionType::get(
      commonTypeAddr, {commonTypeAddr, commonTypeAddr});

  auto allocateCommonType =
      mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {voidPtrType, intType});
  auto allocateTupleType = mlir::LLVM::LLVMFunctionType::get(tupleTypeAddr, {intType});
  auto appendTupleType = mlir::LLVM::LLVMFunctionType::get(intType, {tupleTypeAddr, commonTypeAddr});

  auto deallocateCommonType =
      mlir::LLVM::LLVMFunctionType::get(voidType, commonTypeAddr);

  auto commonBinopType = mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {commonTypeAddr, commonTypeAddr, intType});
  auto commonUnaryopType = mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {commonTypeAddr, intType});

  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "performCommonTypeBINOP",
                                            commonBinopType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "performCommonTypeUNARYOP",
                                          commonUnaryopType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "printCommonType",
                                            printType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "promotion",
                                            promoteCommonTypeFuncType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "allocateCommonType",
                                            allocateCommonType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "allocateTuple",
                                            allocateTupleType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "appendTuple",
                                            appendTupleType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "deallocateCommonType",
                                            deallocateCommonType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "commonTypeToBool", mlir::LLVM::LLVMFunctionType::get(boolType, {commonTypeAddr}));
}

mlir::Value BackEnd::performBINOP(mlir::Value left, mlir::Value right, BINOP op) {
  mlir::LLVM::LLVMFuncOp binopFunc=
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("performCommonTypeBINOP");

  auto result = builder->create<mlir::LLVM::CallOp>(loc, 
      binopFunc, 
      mlir::ValueRange({
        left, 
        right, 
        generateInteger(op)})
      ).getResult();

  std::string newLabel =
      "OBJECT_NUMBER" + std::to_string(this->allocatedObjects);
  this->objectLabels.push_back(newLabel);
  this->generateDeclaration(newLabel, result);
  this->allocatedObjects++;

  return result;
}

mlir::Value BackEnd::promotion(mlir::Value left, mlir::Value right) {
  mlir::LLVM::LLVMFuncOp promotionFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("promotion");

  auto result = builder->create<mlir::LLVM::CallOp>(loc, promotionFunc, mlir::ValueRange({left, right})).getResult();

  // we create a new object, have to tag it
  std::string newLabel =
      "OBJECT_NUMBER" + std::to_string(this->allocatedObjects);
  this->objectLabels.push_back(newLabel);
  this->generateDeclaration(newLabel, result);
  this->allocatedObjects++;

  return result;
}

mlir::Value BackEnd::performUNARYOP(mlir::Value val, UNARYOP op) {
  auto unaryopFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("performCommonTypeUNARYOP");
  auto result = builder->create<mlir::LLVM::CallOp>(loc,
                                                    unaryopFunc,
                                                    mlir::ValueRange({val, generateInteger(op)})
          ).getResult();

  std::string newLabel =
          "OBJECT_NUMBER" + std::to_string(this->allocatedObjects);
  this->objectLabels.push_back(newLabel);
  this->generateDeclaration(newLabel, result);
  this->allocatedObjects++;

  return result;
}


mlir::Value BackEnd::generateCallNamed(std::string signature, std::vector<mlir::Value> arguments) {
  mlir::ArrayRef mlirArguments = arguments;
  mlir::LLVM::LLVMFuncOp function = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(signature);

  return builder->create<mlir::LLVM::CallOp>(loc, function, mlirArguments).getResult();
}

// === === === Printing === === ===

void BackEnd::printCommonType(mlir::Value value) {
  mlir::LLVM::GlobalOp global;

  mlir::LLVM::LLVMFuncOp printVecFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printCommonType");
  builder->create<mlir::LLVM::CallOp>(loc, printVecFunc, value);
}

// === === === TYPEs === === === 

/*
 * Common type is a struct wrapper for all of the types in the compiler
 * The runtime handles all operations.
 */
mlir::Value BackEnd::generateCommonType(mlir::Value value, int type) {
  mlir::Value one = generateInteger(1);
  mlir::LLVM::LLVMFuncOp allocateCommonType =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("allocateCommonType");

  mlir::Value typeValue = generateInteger(type);

  mlir::Value valuePtr = builder->create<mlir::LLVM::AllocaOp>(
      loc, mlir::LLVM::LLVMPointerType::get(value.getType()), one);

  builder->create<mlir::LLVM::StoreOp>(loc, value, valuePtr);

  auto result = builder->create<mlir::LLVM::CallOp>(
      loc, 
      allocateCommonType, 
      mlir::ValueRange({valuePtr, typeValue})
      )
    .getResult();

  std::string newLabel =
      "OBJECT_NUMBER" + std::to_string(this->allocatedObjects);
  this->objectLabels.push_back(newLabel);
  this->generateDeclaration(newLabel, result);
  this->allocatedObjects++;

  return result;
}

mlir::Value BackEnd::generateValue(int value) {
  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getI32Type(), value);
  
  return this->generateCommonType(result, INT);
}

mlir::Value BackEnd::generateValue(bool value) {
  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getI1Type(), value);
  return this->generateCommonType(result, value);
}

mlir::Value BackEnd::generateValue(float value) {
  // google says so. no idea what this is 
  llvm::APFloat floatValue = llvm::APFloat(value);

  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getF32Type(), floatValue);

  return this->generateCommonType(result, REAL);
}

mlir::Value BackEnd::generateValue(char value) {
  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getI8Type(), value);
  return this->generateCommonType(result, CHAR);
}

mlir::Value BackEnd::generateValue(std::vector<mlir::Value> values) {
  mlir::LLVM::LLVMFuncOp allocateTupleFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("allocateTuple");

  auto tuple = builder->create<mlir::LLVM::CallOp>(loc, allocateTupleFunc, mlir::ValueRange({generateInteger((int) values.size())})).getResult();

  mlir::LLVM::LLVMFuncOp appendTuple = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("appendTuple");

  for (auto value : values) {
    builder->create<mlir::LLVM::CallOp>(loc, appendTuple, mlir::ValueRange({tuple, value}));
  }

  return this->generateCommonType(tuple, TUPLE);
}

mlir::Value BackEnd::generateInteger(int value) {
  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getI32Type(), value);
  return result;
}

void BackEnd::deallocateObjects() {
  for (auto label : objectLabels) {
    mlir::LLVM::LLVMFuncOp deallocateObject =
        module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("deallocateCommonType");

    auto object = this->generateLoadIdentifier(label);

    builder->create<mlir::LLVM::CallOp>(loc, deallocateObject, object);
  }
}
// don't need the types for much, just set stuff up so we know what to cast to
mlir::Block* BackEnd::generateFunctionDefinition(std::string signature, size_t argumentSize, bool isVoid) {
    auto currentBlock = builder->getBlock();

    auto intType = builder->getI32Type();
    auto intPtrType = mlir::LLVM::LLVMPointerType::get(intType);
    auto voidType = mlir::LLVM::LLVMVoidType::get(&context);

    auto commonType =
      mlir::LLVM::LLVMStructType::getLiteral(&context, {intType, intPtrType});
    auto commonTypeAddr = mlir::LLVM::LLVMPointerType::get(commonType);

    // don't really need this for "types" since all of our types are the same.
    // however we need the size
    std::vector<mlir::Type> parameters;

    for (int i = 0 ; i < argumentSize ; i ++) {
      parameters.push_back(commonTypeAddr);
    }

    llvm::ArrayRef translatedList = parameters;

    mlir::Type returnType;
    // all return types are common type addresses.
    if (isVoid) {
      returnType = voidType;
    } else {
      returnType = commonTypeAddr;
    }

    auto functionType = mlir::LLVM::LLVMFunctionType::get(returnType, translatedList, false);

    builder->setInsertionPointToStart(module.getBody());

    mlir::LLVM::LLVMFuncOp function = builder->create<mlir::LLVM::LLVMFuncOp>(loc, signature, functionType, ::mlir::LLVM::Linkage::Internal);
    mlir::Block *entry = function.addEntryBlock();
    builder->setInsertionPointToStart(entry);

    return currentBlock;
}

void BackEnd::generateEndFunctionDefinition(mlir::Block* returnBlock) {
    builder->setInsertionPointToEnd(returnBlock);
}

void BackEnd::generateReturn(mlir::Value returnVal) {
  builder->create<mlir::LLVM::ReturnOp>(loc, returnVal);
}

/**
 * Generate a vector of the given size
 */
mlir::Value BackEnd::generateVectorOfSize(mlir::Value sizeOf) {
  mlir::LLVM::LLVMFuncOp getTrueSizeFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("getTrueVectorSize");
  mlir::LLVM::LLVMFuncOp makeVector =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("allocateVector");

  mlir::Value trueSize =
      builder->create<mlir::LLVM::CallOp>(loc, getTrueSizeFunc, sizeOf)
          .getResult();

  mlir::Value structAddr =
      builder->create<mlir::LLVM::CallOp>(loc, makeVector, trueSize)
          .getResult();

  // deallocation hack 2025
  std::string newLabel =
      "VECTOR_NUMBER_" + std::to_string(this->allocatedVectors);
  this->allocatedVectors++;
  this->vectorLabels.push_back(newLabel);
  this->generateDeclaration(newLabel, structAddr);

  return structAddr;
}

/**
 * lower is an integer value
 * upper is an integer value
 * Generate a vector from a range. [1..2]
 * */
mlir::Value BackEnd::generateVectorFromRange(mlir::Value lower,
                                             mlir::Value upper) {
  mlir::Value sizeOf =
      this->generateIntegerBinaryOperation(upper, lower, BINOP::SUB);
  auto intType = builder->getI32Type();
  mlir::Value one = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getIntegerAttr(intType, 1));
  sizeOf = this->generateIntegerBinaryOperation(one, sizeOf, BINOP::ADD);

  auto structAddr = this->generateVectorOfSize(sizeOf);

  auto fillFunction = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("fill");
  builder->create<mlir::LLVM::CallOp>(
      loc, fillFunction, mlir::ValueRange({structAddr, lower, upper}));

  return structAddr;
}

/**
 * @param left
 * @param right
 * @param op
 * @return the result of the OP
 */
mlir::Value BackEnd::generateIntegerBinaryOperation(mlir::Value left,
                                                    mlir::Value right,
                                                    BINOP op) {
  mlir::Value result;

  switch (op) {
  case ADD:
    result = builder->create<mlir::LLVM::AddOp>(loc, left, right);
    break;
  case SUB:
    result = builder->create<mlir::LLVM::SubOp>(loc, left, right);
    break;
  case MULT:
    result = builder->create<mlir::LLVM::MulOp>(loc, left, right);
    break;
  case DIV:
    result = builder->create<mlir::LLVM::SDivOp>(loc, left, right);
    break;
  case EQUAL:
    result = builder->create<mlir::LLVM::ICmpOp>(
        loc, builder->getI1Type(), mlir::LLVM::ICmpPredicate::eq, left, right);
    break;
  case NEQUAL:
    result = builder->create<mlir::LLVM::ICmpOp>(
        loc, builder->getI1Type(), mlir::LLVM::ICmpPredicate::ne, left, right);
    break;
  case GTHAN:
    result = builder->create<mlir::LLVM::ICmpOp>(
        loc, builder->getI1Type(), mlir::LLVM::ICmpPredicate::sgt, left, right);
    break;
  case LTHAN:
    result = builder->create<mlir::LLVM::ICmpOp>(
        loc, builder->getI1Type(), mlir::LLVM::ICmpPredicate::slt, left, right);
    break;
  }

  // zero extension from I1-I32. LLVM picky about types
  result =
      builder->create<mlir::LLVM::ZExtOp>(loc, builder->getI32Type(), result);
  return result;
}


void BackEnd::generateDeclaration(std::string varName, mlir::Value value) {
  this->generateInitializeGlobalVar(varName, value);
  this->generateAssignment(varName, value);
}

void BackEnd::generateAssignment(std::string varName, mlir::Value value) {
  mlir::LLVM::GlobalOp global;

  if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(varName))) {
    llvm::errs() << "Referencing undefined variable";
    return;
  }

  mlir::Value globalPtr = builder->create<mlir::LLVM::AddressOfOp>(loc, global);
  builder->create<mlir::LLVM::StoreOp>(loc, value, globalPtr);
}

void BackEnd::generateInitializeGlobalVar(std::string varName,
                                          mlir::Value value) {
  mlir::Block *currentBlock =
      builder->getBlock(); // get current block we are in so we can return here
                           // later
  builder->setInsertionPointToStart(
      module
          .getBody()); // set entrypoint to modulOP to declare global var there
  // first element of domain vector to later store to the domainVar at global
  // initiallize domainVar at global variable
  builder->create<mlir::LLVM::GlobalOp>(
      loc, value.getType(),
      /*isconstant */ false, ::mlir::LLVM::Linkage::Internal, varName,
      builder->getZeroAttr(value.getType()),
      /*alignment*/ 0);

  builder->setInsertionPointToEnd(
      currentBlock); // set insertionpoint back to the end of current to
                     // continue the flow: NEED THIS
}

mlir::Value BackEnd::generateLoadIdentifier(std::string varName) {
  mlir::Value globalPtr = this->generateLoadIdentifierPtr(varName);
  mlir::Value value = builder->create<mlir::LLVM::LoadOp>(loc, globalPtr);
  return value;
}

mlir::Value BackEnd::generateLoadArgument(size_t index) {
  auto val = builder->getBlock()->getArguments().vec()[index];
  return val;
}
/*
 * used to store the comparison result to an address so we can load it later in
 * an another block for looping
 *
 */
mlir::Value BackEnd::generateStoreConstant(mlir::Value value) {
  mlir::Value one =
      builder->create<mlir::LLVM::ConstantOp>(loc, builder->getI32Type(), 1);
  mlir::Value addr = builder->create<mlir::LLVM::AllocaOp>(
      loc, mlir::LLVM::LLVMPointerType::get(value.getType()), one, 0);
  builder->create<mlir::LLVM::StoreOp>(loc, value,
                                       addr); // store value to the addr
  return addr;
}

/*
 * generates unconditional jump to block:
 * Br  block
 */
void BackEnd::generateEnterBlock(mlir::Block *block) {
  // enter this block
  builder->create<mlir::LLVM::BrOp>(loc, block);
}

mlir::Block *BackEnd::generateBlock() {
  mlir::Block *newBlock = mainFunc.addBlock();
  return newBlock;
}

/*
 * Jumps to the true block or false block, depending on the value of `cmpVal`
 */
void BackEnd::generateCompAndJump(mlir::Block *trueBlock,
                                  mlir::Block *falseBlock, mlir::Value cmpVal) {
  // jump depending on the value of cmpVal
  builder->create<mlir::LLVM::CondBrOp>(loc, cmpVal, trueBlock, falseBlock);
}

void BackEnd::setBuilderInsertionPoint(
    mlir::Block *block) { // set insertion point but no smart pointer:
  builder->setInsertionPointToEnd(block);
}

mlir::Value BackEnd::generateIndexWithInteger(mlir::Value vector,
                                              mlir::Value index) {
  // result is sizeof indexor
  auto vectorToIntegerIndex =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("vectorToIntegerIndex");
  return builder
      ->create<mlir::LLVM::CallOp>(loc, vectorToIntegerIndex,
                                   mlir::ValueRange({vector, index}))
      .getResult();
}

/*
 * Given an MLIR Value of a commonType,
 * returns an MLIR value of the downcasted boolean value as an i1 type
 */
mlir::Value BackEnd::downcastToBool(mlir::Value val) {
  auto downcastFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("commonTypeToBool");
  return builder->create<mlir::LLVM::CallOp>(loc,
                                             downcastFunc,
                                             mlir::ValueRange({val})
  ).getResult();
}

/*
 * @param: domainVecAddr
 * size = domainVecAddr->vectorObj->size
 * Global domainVar = *(rangeVecAddr->vector)   // need to define this in global
 * variable i = 0  // this needs to be malloced cuz we cannot changed the
 * constant once declaed
 */
mlir::Value BackEnd::generateGeneratorBegin(mlir::Value domainVecAddr,
                                            std::string domainVar) {
  mlir::Type intType = builder->getI32Type();
  mlir::Value zero = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getIntegerAttr(intType, 0));

  return zero;
}

mlir::Value BackEnd::getVectorSize(mlir::Value vectorAddr) {
  mlir::Type intType = builder->getI32Type();
  mlir::Value zero = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getIntegerAttr(intType, 0));
  mlir::Value one = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getIntegerAttr(intType, 1));

  mlir::Value sizeAddr = builder->create<mlir::LLVM::GEPOp>(
      loc, mlir::LLVM::LLVMPointerType::get(intType), vectorAddr,
      mlir::ValueRange({zero, one}));
  mlir::Value size = builder->create<mlir::LLVM::LoadOp>(loc, sizeAddr);
  return size;
}

void BackEnd::generateStoreValueInVector(mlir::Value vectorAddr,
                                         mlir::Value index,
                                         mlir::Value exprResult) {
  // store
  auto vectorStoreValueAtIndexFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("vectorStoreValueAtIndex");
  builder->create<mlir::LLVM::CallOp>(
      loc, vectorStoreValueAtIndexFunc,
      mlir::ValueRange({vectorAddr, index, exprResult}));
  // increment i;
}

/*
 * @param: indexAddr
 * load i, (indexArr)
 * i = i + 1
 * store i, (indexArr)
 */
void BackEnd::generateIncrementIndex(mlir::Value indexAddr) {
  mlir::Value one = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getIntegerAttr(builder->getI32Type(), 1));
  mlir::Value index = builder->create<mlir::LLVM::LoadOp>(loc, indexAddr);
  mlir::Value nextIndex =
      builder->create<mlir::LLVM::AddOp>(loc, index, one); // i = i + 1;

  builder->create<mlir::LLVM::StoreOp>(
      loc, nextIndex, indexAddr); // upadte the indexAddr to next in
}

/*
 * @param: domainVecAddr: address to the domain vector
 * @indexAddr: index addr
 * @domainVar: the domain variable already declared in global
 *
 * update the domainVar, whose declared on global, to the next value taken from
 * rangevector. i.e @DomainVar = rangeVec[++i]
 */
void BackEnd::generateUpdateDomainVar(mlir::Value domainVecAddr,
                                      mlir::Value indexAddr,
                                      std::string domainVar) {
  mlir::Value index = builder->create<mlir::LLVM::LoadOp>(loc, indexAddr);
  auto vectorLoadValueAtIndexFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("vectorLoadValueAtIndex");
  mlir::Value elementAtIndex =
      builder
          ->create<mlir::LLVM::CallOp>(loc, vectorLoadValueAtIndexFunc,
                                       mlir::ValueRange({domainVecAddr, index}))
          .getResult();
  this->generateAssignment(
      domainVar, elementAtIndex); // update the domainVar to hold the next
}

mlir::Value BackEnd::generateValuePtr(mlir::Value value) {
  mlir::Value one = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getIntegerAttr(builder->getI32Type(), 1));
  mlir::Value indexAddr = builder->create<mlir::LLVM::AllocaOp>(
      loc, mlir::LLVM::LLVMPointerType::get(value.getType()), one, 0);

  builder->create<mlir::LLVM::StoreOp>(loc, value, indexAddr);
  return indexAddr;
}

mlir::Value BackEnd::generateLoadIdentifierPtr(std::string varName) {
  mlir::LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(varName))) {
    llvm::errs() << "Storage fail!";
    exit(1);
  }
  mlir::Value globalPtr = builder->create<mlir::LLVM::AddressOfOp>(loc, global);
  return globalPtr;
}

mlir::Value BackEnd::generateLoadValue(mlir::Value addr) {
  return builder->create<mlir::LLVM::LoadOp>(loc, addr);
}

void BackEnd::generateSetVectorSize(mlir::Value vecAddr, mlir::Value size) {
  mlir::Type intType = builder->getI32Type();
  mlir::Value zero = generateInteger(0);
  mlir::Value one = generateInteger(1);
  mlir::Value sizeAddr = builder->create<mlir::LLVM::GEPOp>(
      loc, mlir::LLVM::LLVMPointerType::get(intType), vecAddr,
      mlir::ValueRange({zero, one}));

  builder->create<mlir::LLVM::StoreOp>(loc, size, sizeAddr);
}
