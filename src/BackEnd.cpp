#include "llvm/ADT/APFloat.h"
#include "llvm/IR/Attributes.h"
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

  auto a = this->generateCommonType(generateValue(3.2f), 3);
  auto b = this->generateCommonType(generateValue('c'), 2);
  auto c = this->generateCommonType(generateValue(1), 1);

  this->printCommonType(a);
  this->printCommonType(b);
  this->printCommonType(c);

  this->deallocateVectors();

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

  setupPrint();
  setupVectorRuntime();
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
  auto intType = builder->getI32Type();
  auto int64Type = builder->getI64Type();

  auto intPtrType = mlir::LLVM::LLVMPointerType::get(intType);

  // mlir doesn't allow void types. we do a little hacking
  auto voidPtrType = mlir::LLVM::LLVMPointerType::get(int64Type);

  auto commonType =
      mlir::LLVM::LLVMStructType::getLiteral(&context, {intPtrType, intType});

  auto commonTypeAddr = mlir::LLVM::LLVMPointerType::get(commonType);

  auto printType = mlir::LLVM::LLVMFunctionType::get(
      voidType, {commonTypeAddr});
  auto allocateCommonType =
      mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {voidPtrType, intType});
  auto deallocateCommonType =
      mlir::LLVM::LLVMFunctionType::get(voidType, commonTypeAddr);

  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "printCommonType",
                                            printType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "allocateCommonType",
                                            allocateCommonType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "deallocateCommonType",
                                            deallocateCommonType);
}


/**
 * Set up the function signature for the runtime function
 * we call dynamic library during runtime
 */
void BackEnd::setupVectorRuntime() {
  auto voidType = mlir::LLVM::LLVMVoidType::get(&context);
  auto intType = builder->getI32Type();

  auto intPtrType = mlir::LLVM::LLVMPointerType::get(intType);

  auto vectorStruct =
      mlir::LLVM::LLVMStructType::getLiteral(&context, {intPtrType, intType});
  auto vectorStructAddr = mlir::LLVM::LLVMPointerType::get(vectorStruct);

  auto allocateVectorType =
      mlir::LLVM::LLVMFunctionType::get(vectorStructAddr, intType);
  auto deallocateVectorType =
      mlir::LLVM::LLVMFunctionType::get(voidType, vectorStructAddr);

  auto v2iindexType =
      mlir::LLVM::LLVMFunctionType::get(intType, {vectorStructAddr, intType});
  auto v2vindexType = mlir::LLVM::LLVMFunctionType::get(
      voidType, {vectorStructAddr, vectorStructAddr, vectorStructAddr});
  auto v2vbinopType = mlir::LLVM::LLVMFunctionType::get(
      voidType,
      {vectorStructAddr, vectorStructAddr, vectorStructAddr, intType});
  auto v2ibinopType = mlir::LLVM::LLVMFunctionType::get(
      voidType, {vectorStructAddr, intType, vectorStructAddr, intType});
  auto i2vbinopType = mlir::LLVM::LLVMFunctionType::get(
      voidType, {intType, vectorStructAddr, vectorStructAddr, intType});

  auto getSizeType = mlir::LLVM::LLVMFunctionType::get(
      intType, {vectorStructAddr, vectorStructAddr});
  auto getTrueVectorSizeType =
      mlir::LLVM::LLVMFunctionType::get(intType, intType);
  auto fillType = mlir::LLVM::LLVMFunctionType::get(voidType, vectorStructAddr);
  auto printType = mlir::LLVM::LLVMFunctionType::get(
      voidType, {vectorStructAddr, intType, intType});
  auto vectorStoreValueType = mlir::LLVM::LLVMFunctionType::get(
      voidType, {vectorStructAddr, intType, intType});
  auto vectorLoadValueType =
      mlir::LLVM::LLVMFunctionType::get(intType, {vectorStructAddr, intType});

  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "allocateVector",
                                          allocateVectorType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "deallocateVector",
                                          deallocateVectorType);

  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "getMaxSize", getSizeType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "getTrueVectorSize",
                                          getTrueVectorSizeType);

  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "vectorToIntegerIndex",
                                          v2iindexType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "vectorToVectorIndex",
                                          v2vindexType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "vectorToVectorBINOP",
                                          v2vbinopType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "integerToVectorBINOP",
                                          i2vbinopType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "vectorToIntegerBINOP",
                                          v2ibinopType);

  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "fill", printType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "printVec", fillType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "vectorStoreValueAtIndex",
                                          vectorStoreValueType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "vectorLoadValueAtIndex",
                                          vectorLoadValueType);
}

/**
 * Set up the function signature for the runtime function
 * we call dynamic library during runtime
 */
void BackEnd::setupPrint() {
  auto voidType = mlir::LLVM::LLVMVoidType::get(&context);
  mlir::Type intType = mlir::IntegerType::get(&context, 32);
  auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(voidType, intType);

  // Insert the printf function into the body of the parent module.
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "print", llvmFnType);
}

/**
 * Calls the runtime "print" signature
 * @param value
 */
void BackEnd::print(mlir::Value value) {
  mlir::LLVM::GlobalOp global;

  mlir::LLVM::LLVMFuncOp printfFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("print");
  builder->create<mlir::LLVM::CallOp>(loc, printfFunc, value);
}

void BackEnd::printVec(mlir::Value value) {
  mlir::LLVM::GlobalOp global;

  mlir::LLVM::LLVMFuncOp printVecFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printVec");
  builder->create<mlir::LLVM::CallOp>(loc, printVecFunc, value);
}

// === === === Printing === === ===

void BackEnd::printCommonType(mlir::Value value) {
  mlir::LLVM::GlobalOp global;

  mlir::LLVM::LLVMFuncOp printVecFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printCommonType");
  builder->create<mlir::LLVM::CallOp>(loc, printVecFunc, value);
}

// === === === TYPEs === === === 

mlir::Value BackEnd::generateCommonType(mlir::Value value, int type) {
  auto intType = builder->getI32Type();
  mlir::Value one = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getIntegerAttr(intType, 1));

  mlir::LLVM::LLVMFuncOp allocateCommonType =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("allocateCommonType");
    mlir::Value typeValue = generateValue(type);

  mlir::Value valuePtr = builder->create<mlir::LLVM::AllocaOp>(
      loc, mlir::LLVM::LLVMPointerType::get(typeValue.getType()), one, 0);

  builder->create<mlir::LLVM::StoreOp>(loc, value, valuePtr);

  return builder->create<mlir::LLVM::CallOp>(
      loc, 
      allocateCommonType, 
      mlir::ValueRange({valuePtr, typeValue})
      )
    .getResult();
}


mlir::Value BackEnd::generateInteger(int value) {
  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getI32Type(), value);
  return result;
}

mlir::Value BackEnd::generateValue(int value) {
  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getI32Type(), value);
  return result;
}

mlir::Value BackEnd::generateValue(float value) {
  // google says so. no idea what this is 
  llvm::APFloat floatValue = llvm::APFloat(value);

  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getF32Type(), floatValue);

  return result;
}

mlir::Value BackEnd::generateValue(char value) {
  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getI8Type(), value);
  return result;
}

/**
 * Dirty free trick
 * */
void BackEnd::deallocateVectors() {
  for (auto label : vectorLabels) {
    mlir::LLVM::LLVMFuncOp deallocateVectorFunc =
        module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("deallocateVector");
    auto vector = this->generateLoadIdentifier(label);
    builder->create<mlir::LLVM::CallOp>(loc, deallocateVectorFunc, vector);
  }
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

mlir::Value BackEnd::generateVectorToVectorBINOP(mlir::Value left,
                                                 mlir::Value right, BINOP op) {
  auto result = this->generateVectorToFit(left, right);

  auto vectorToVectorFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("vectorToVectorBINOP");
  auto binop = this->generateInteger(op);

  builder->create<mlir::LLVM::CallOp>(
      loc, vectorToVectorFunc, mlir::ValueRange({left, right, result, binop}));

  return result;
}

mlir::Value BackEnd::generateIntegerToVectorBINOP(mlir::Value left,
                                                  mlir::Value right, BINOP op) {
  auto result = this->generateVectorToFit(right, right);

  auto vectorToVectorFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("integerToVectorBINOP");
  auto binop = this->generateInteger(op);

  builder->create<mlir::LLVM::CallOp>(
      loc, vectorToVectorFunc, mlir::ValueRange({left, right, result, binop}));

  return result;
}

mlir::Value BackEnd::generateVectorToIntegerBINOP(mlir::Value left,
                                                  mlir::Value right, BINOP op) {
  auto result = this->generateVectorToFit(left, left);

  auto vectorToVectorFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("vectorToIntegerBINOP");
  auto binop = this->generateInteger(op);

  builder->create<mlir::LLVM::CallOp>(
      loc, vectorToVectorFunc, mlir::ValueRange({left, right, result, binop}));

  return result;
}

/**
 * left is a vector struct address type
 * right is a vector struct address type
 * we are doing operations with two vectors. The size depends on both of them
 **/
mlir::Value BackEnd::generateVectorToFit(mlir::Value left, mlir::Value right) {
  auto getSizeFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("getMaxSize");
  auto newSize = builder
                     ->create<mlir::LLVM::CallOp>(
                         loc, getSizeFunc, mlir::ValueRange({left, right}))
                     .getResult();

  return this->generateVectorOfSize(newSize);
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

void BackEnd::generateCompAndJump(mlir::Block *trueBlock,
                                  mlir::Block *falseBlock, mlir::Value addr) {
  // load from addr, do icmp, and jump
  mlir::Value zero =
      builder->create<mlir::LLVM::ConstantOp>(loc, builder->getI32Type(), 0);
  mlir::Value cmpResult = builder->create<mlir::LLVM::ICmpOp>(
      loc, builder->getI1Type(), mlir::LLVM::ICmpPredicate::ne, addr, zero);
  builder->create<mlir::LLVM::CondBrOp>(loc, cmpResult, trueBlock, falseBlock);
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

/**
 * Index vector
 * */
mlir::Value BackEnd::generateIndexWithVector(mlir::Value indexee,
                                             mlir::Value indexor) {
  // result is sizeof indexor
  auto result = this->generateVectorToFit(indexor, indexor);
  auto vectorToVectorFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("vectorToVectorIndex");
  builder->create<mlir::LLVM::CallOp>(
      loc, vectorToVectorFunc, mlir::ValueRange({indexee, indexor, result}));

  return result;
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
