#include "llvm/ADT/APFloat.h"
#include "CompileTimeExceptions.h"
#include "Types/TYPES.h"
#include "Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include <assert.h>
#include <iostream>
#include <stdexcept>
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

//#define DEBUG

/**
 *  Set up main function
 */
void BackEnd::init() {
#ifdef DEBUG
  std::cout << "BEGINNING CODEGEN" << std::endl;
#endif

  auto intType = builder->getI32Type();
  auto mainType = mlir::LLVM::LLVMFunctionType::get(intType, {}, false);

  auto mainFunc = builder->create<mlir::LLVM::LLVMFuncOp>(loc, "main", mainType);

  // override handler to not output to stderr
   context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
     return;
   });

  functionStack.push_back(mainFunc);
  mainEntry = mainFunc.addEntryBlock();
  builder->setInsertionPointToStart(mainEntry);
  this->pushScope();
}

void BackEnd::verifyFunction(int line, std::string name) {
  if (mlir::verify(functionStack[functionStack.size()-1]).failed()) {
    throw ReturnError(line, name + " does not have a return statement reachable by all control flows");
  }
  functionStack.pop_back();
}
/**
 * Finish codegen + main function.
 */
void BackEnd::generate() {
#ifdef DEBUG
  std::cout << "CODEGEN FINISHED, ending main function and outputting"
            << std::endl;
#endif
  std::vector<mlir::Value> mainArgs;

  this->generateCallNamed("main", mainArgs);
  this->deallocateObjects();
  this->popScope();

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
#ifdef DEBUG
  std::cout << "==============================================================="
               "=================\n";
  std::cout << "LLVM IR\n";
  std::cout << "-------\n";
  llvmModule->dump();
  std::cout << "==============================================================="
               "=================\n";
#endif
  // print LLVM IR to file
  llvm::raw_os_ostream output(this->out);
  output << *llvmModule;

  return 0;
}

void BackEnd::functionShowcase() {
    // run getStreamState with streamStatePtr
    auto streamState = module.lookupSymbol<mlir::LLVM::GlobalOp>("streamState");

    // MLIR doesn't allow direct access to globals, so we have to use an addressof
    auto streamStatePtr = builder->create<mlir::LLVM::AddressOfOp>(loc, streamState);

    // lookup getStreamState runtime function
    auto getStreamStateFunc= module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("getStreamState");
    builder->create<mlir::LLVM::CallOp>(loc, getStreamStateFunc, mlir::ValueRange({streamStatePtr}));
    builder->create<mlir::LLVM::CallOp>(loc, getStreamStateFunc, mlir::ValueRange({streamStatePtr}));
}

BackEnd::BackEnd(std::ostream &out)
    : out(out), loc(mlir::UnknownLoc::get(&context)) {
  context.loadDialect<mlir::LLVM::LLVMDialect>();

  builder = std::make_shared<mlir::OpBuilder>(&context);

  // Open a new context and module.
  module = mlir::ModuleOp::create(builder->getUnknownLoc());
  builder->setInsertionPointToStart(module.getBody());

  setupCommonTypeRuntime();
  setupStreamRuntime();
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
  
  auto listType =
      mlir::LLVM::LLVMStructType::getLiteral(&context, {intType, intType, mlir::LLVM::LLVMPointerType::get(commonTypeAddr)});
  auto listTypeAddr = mlir::LLVM::LLVMPointerType::get(listType);

  auto printType = mlir::LLVM::LLVMFunctionType::get(
      voidType, {commonTypeAddr});
  auto streamInType = mlir::LLVM::LLVMFunctionType::get(
      voidType, {commonTypeAddr, intPtrType});
  auto allocateCommonType =
      mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {voidPtrType, intType});
  auto allocateFromRange =
      mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {commonTypeAddr, commonTypeAddr});

  auto allocateListType = mlir::LLVM::LLVMFunctionType::get(listTypeAddr, {intType});
  auto allocateListFromCommon = mlir::LLVM::LLVMFunctionType::get(listTypeAddr, {commonTypeAddr});
  auto appendListType = mlir::LLVM::LLVMFunctionType::get(intType, {listTypeAddr, commonTypeAddr});
  auto appendCommon = mlir::LLVM::LLVMFunctionType::get(intType, {commonTypeAddr, commonTypeAddr});
  auto normalize = mlir::LLVM::LLVMFunctionType::get(intType, {commonTypeAddr});

  auto indexCommonType = mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {commonTypeAddr, commonTypeAddr});
  auto deallocateCommonType =
      mlir::LLVM::LLVMFunctionType::get(voidType, commonTypeAddr);
  auto commonCastType = mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {commonTypeAddr, intType});
  auto commonReferenceAssign = mlir::LLVM::LLVMFunctionType::get(voidType, {commonTypeAddr, commonTypeAddr});
  auto copy= mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {commonTypeAddr});

  auto commonBinopType = mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {commonTypeAddr, commonTypeAddr, intType});
  auto commonUnaryopType = mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {commonTypeAddr, intType});

  auto lengthType = mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {commonTypeAddr});
  // setup runtime stream_state function
  auto streamStateFunctionType = mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {intPtrType});
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "normalize",
                                            normalize);

  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "__rows",
                                            lengthType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "copyCommonType", copy);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "__columns",
                                            lengthType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "__length",
                                            lengthType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "__reverse", lengthType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "__format", lengthType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "__stream_state", streamStateFunctionType);

  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "indexCommonType",
                                            indexCommonType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "assignByReference",
                                            commonReferenceAssign);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "performCommonTypeBINOP",
                                            commonBinopType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "performCommonTypeUNARYOP",
                                          commonUnaryopType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "printCommonType",
                                            printType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "streamOut",
                                          printType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "streamIn", streamInType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "castHelper",
                                          commonCastType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "cast",
                                          allocateFromRange);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "allocateCommonType",
                                            allocateCommonType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "allocateList",
                                            allocateListType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "allocateListFromCommon",
                                            allocateListFromCommon);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "allocateFromRange",
                                            allocateFromRange);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "appendList",
                                            appendListType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "appendCommon",
                                            appendCommon);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "deallocateCommonType",
                                            deallocateCommonType);
  builder->create<mlir::LLVM::LLVMFuncOp>(loc, "commonTypeToBool", mlir::LLVM::LLVMFunctionType::get(boolType, {commonTypeAddr}));

  // builtin functions
  // TODO: delete "silly" function once we have a proper stdlib
//   builder->create<mlir::LLVM::LLVMFuncOp>(loc, "__silly", mlir::LLVM::LLVMFunctionType::get(commonTypeAddr, {commonTypeAddr}));
}

void BackEnd::setupStreamRuntime() {
  // setup a global (int) variable called streamState
  auto intType = builder->getI32Type();
  auto intPtrType = mlir::LLVM::LLVMPointerType::get(intType);
  auto voidType = mlir::LLVM::LLVMVoidType::get(&context);

  // streamState variable
  auto streamState = builder->create<mlir::LLVM::GlobalOp>(
          loc, intType, false, mlir::LLVM::Linkage::Internal,
          "streamState", builder->getIntegerAttr(intType, 0));
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


  auto newLabel = trackObject();
  this->generateDeclaration(newLabel, result);

  return result;
}

std::string BackEnd::trackObject() {
  std::string newLabel =
      "OBJECT_NUMBER" + std::to_string(this->allocatedObjects);
  std::vector<std::string> *currentScope  = *(this->objectLabels.end() - 1);
  currentScope->push_back(newLabel);
  this->allocatedObjects++;
  return newLabel;
}

void BackEnd::pushScope() {
  std::vector<std::string> *labelScope = new std::vector<std::string>();
  this->objectLabels.push_back(labelScope);
}

void BackEnd::popScope() {
  this->objectLabels.pop_back();
}

mlir::Value BackEnd::indexCommonType(mlir::Value indexee, mlir::Value indexor) {
  mlir::LLVM::LLVMFuncOp promotionFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("indexCommonType");

  return builder->create<mlir::LLVM::CallOp>(loc, promotionFunc, mlir::ValueRange({indexee, indexor})).getResult();
}

mlir::Value BackEnd::copyCommonType(mlir::Value val) {
  mlir::LLVM::LLVMFuncOp copyFunc=
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("copyCommonType");

  return builder->create<mlir::LLVM::CallOp>(loc, copyFunc, mlir::ValueRange({val})).getResult();
}

void BackEnd::normalize(mlir::Value matrix) {
  mlir::LLVM::LLVMFuncOp normalizeFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("normalize");

  builder->create<mlir::LLVM::CallOp>(loc, normalizeFunc, mlir::ValueRange({matrix}));
}

mlir::Value BackEnd::cast(mlir::Value left, mlir::Value right) {
  mlir::LLVM::LLVMFuncOp promotionFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("cast");

  auto result = builder->create<mlir::LLVM::CallOp>(loc, promotionFunc, mlir::ValueRange({left, right})).getResult();

  // we create a new object, have to tag it
  auto newLabel = trackObject();
  this->generateDeclaration(newLabel, result);
  this->allocatedObjects++;

  return result;
}
mlir::Value BackEnd::cast(mlir::Value left, TYPE toType) {
  mlir::LLVM::LLVMFuncOp promotionFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("castHelper");

  auto result = builder->create<mlir::LLVM::CallOp>(loc, promotionFunc, mlir::ValueRange({left, this->generateInteger(toType)})).getResult();

  // we create a new object, have to tag it
  auto newLabel = trackObject();
  this->generateDeclaration(newLabel, result);
  this->allocatedObjects++;

  return result;
}

/**
 * append to a list
 **/
void BackEnd::appendCommon(mlir::Value destination, mlir::Value item) {
  mlir::LLVM::LLVMFuncOp appendFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("appendCommon");

  builder->create<mlir::LLVM::CallOp>(loc, appendFunc, mlir::ValueRange({destination, item}));

  return;
}

/*
 * Takes a nullable std::shared_ptr<Type>, and cast the value to that type
 *
 * ONLY WORKS for simple types (integer, char, bool, real)
 * if we pass a vector type, we will just return the value
 * If the type is null, return the same value (no-op)
 */
mlir::Value BackEnd::possiblyCast(mlir::Value val, std::shared_ptr<Type> nullableType) {
  if (nullableType) {
    std::vector<TYPE> acceptableTypes = {TYPE::INTEGER, TYPE::CHAR, TYPE::BOOLEAN, TYPE::REAL};
    if (std::find(acceptableTypes.begin(), acceptableTypes.end(), nullableType->baseTypeEnum) == acceptableTypes.end()) {
      return val;
    } else {
      return cast(val, nullableType->baseTypeEnum);
    }
  } else {
    return val;
  }
}

mlir::Value BackEnd::performUNARYOP(mlir::Value val, UNARYOP op) {
  auto unaryopFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("performCommonTypeUNARYOP");
  auto result = builder->create<mlir::LLVM::CallOp>(loc,
                                                    unaryopFunc,
                                                    mlir::ValueRange({val, generateInteger(op)})
          ).getResult();

  std::string newLabel = trackObject();
  this->generateDeclaration(newLabel, result);

  return result;
}


mlir::Value BackEnd::generateCallNamed(std::string signature, std::vector<mlir::Value> arguments) {
  mlir::ArrayRef mlirArguments = arguments;
  mlir::LLVM::LLVMFuncOp function = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__"+signature);

  auto result = builder->create<mlir::LLVM::CallOp>(loc, function, mlirArguments).getResult();

  this->generateDeclaration(trackObject(), result);

  return result;
}

// === === === Printing === === ===

void BackEnd::printCommonType(mlir::Value value) {
  mlir::LLVM::GlobalOp global;

  mlir::LLVM::LLVMFuncOp printVecFunc =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printCommonType");
  builder->create<mlir::LLVM::CallOp>(loc, printVecFunc, value);
}

/*
 * Functions like printCommonType, but follows the rules for streamOut
 * Such as, no trailing whitespace
 */
void BackEnd::streamOut(mlir::Value value) {
  mlir::LLVM::LLVMFuncOp streamOutFunc =
          module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("streamOut");
  builder->create<mlir::LLVM::CallOp>(loc, streamOutFunc, value);
}

/*
 * Returns (pointer to) global streamState variable (pointer to int)
 */
mlir::Value BackEnd::getStreamStateVar() {
  // run getStreamState with streamStatePtr
  auto streamState = module.lookupSymbol<mlir::LLVM::GlobalOp>("streamState");
  // MLIR doesn't allow direct access to globals, so we have to use an addressof
  auto streamStatePtr = builder->create<mlir::LLVM::AddressOfOp>(loc, streamState);
  return streamStatePtr.getResult();
}

/*
 * Reads from stdin (based on the type of the value) and assigns
 */
void BackEnd::streamIn(mlir::Value value) {
    auto streamStatePtr = getStreamStateVar();
    mlir::LLVM::LLVMFuncOp streamInFunc =
            module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("streamIn");
    builder->create<mlir::LLVM::CallOp>(loc, streamInFunc, mlir::ValueRange({value, streamStatePtr}));
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

  std::string newLabel = trackObject();
  this->generateDeclaration(newLabel, result);

  return result;
}

mlir::Value BackEnd::generateValue(int value) {
  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getI32Type(), value);
  
  return this->generateCommonType(result, INTEGER);
}

mlir::Value BackEnd::generateValue(bool value) {
  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getI1Type(), value);
  return this->generateCommonType(result, BOOLEAN);
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

mlir::Value BackEnd::generateValue(std::string value) {
  std::vector<mlir::Value> values;

  for (char character : value) {
    values.push_back(this->generateValue(character));
  }

  mlir::LLVM::LLVMFuncOp allocateListFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("allocateList");

  auto string = builder->create<mlir::LLVM::CallOp>(loc, allocateListFunc, mlir::ValueRange({generateInteger((int) values.size())})).getResult();

  mlir::LLVM::LLVMFuncOp appendList = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("appendList");

  for (auto commonChar : values) {
    builder->create<mlir::LLVM::CallOp>(loc, appendList, mlir::ValueRange({string, commonChar}));
  }

  return this->generateCommonType(string, STRING);
}

/**
 * empty vector so we can append to it. For filters and generators
 * */
mlir::Value BackEnd::generateValue(mlir::Value length, bool isString) {
  mlir::LLVM::LLVMFuncOp allocateListFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("allocateListFromCommon");
  auto list = builder->create<mlir::LLVM::CallOp>(loc, allocateListFunc, mlir::ValueRange({length})).getResult();

  return this->generateCommonType(list, isString ? STRING : VECTOR);
}

/**
 * range thingy
 * */
mlir::Value BackEnd::generateValue(mlir::Value lower, mlir::Value upper) {
  mlir::LLVM::LLVMFuncOp allocateListFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("allocateFromRange");
  return builder->create<mlir::LLVM::CallOp>(loc, allocateListFunc, mlir::ValueRange({lower, upper})).getResult();
}

mlir::Value BackEnd::generateValue(std::vector<mlir::Value> values) {
  mlir::LLVM::LLVMFuncOp allocateListFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("allocateList");

  auto tuple = builder->create<mlir::LLVM::CallOp>(loc, allocateListFunc, mlir::ValueRange({generateInteger((int) values.size())})).getResult();

  mlir::LLVM::LLVMFuncOp appendTuple = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("appendList");

  for (auto value : values) {
    builder->create<mlir::LLVM::CallOp>(loc, appendTuple, mlir::ValueRange({tuple, value}));
  }

  return this->generateCommonType(tuple, VECTOR);
}

mlir::Value BackEnd::generateInteger(int value) {
  mlir::Value result = builder->create<mlir::LLVM::ConstantOp>(
      loc, builder->getI32Type(), value);
  return result;
}

mlir::Value BackEnd::generateNullValue(std::shared_ptr<Type> type) {
  switch (type->baseTypeEnum) {
    case BOOLEAN:
      return this->generateValue(false);
    case CHAR:
      return this->generateValue((char)0x00);
    case INTEGER:
      return this->generateValue(0);
    case REAL:
      return this->generateValue(0.0f);
    case TUPLE:
      {
        std::vector<mlir::Value> children;
        for (auto childType : type->tupleChildType) {
          children.push_back(generateNullValue(childType.second));
        }

        // magic code
        return this->generateValue(children);
      }
    default:
      throw std::runtime_error("Identity not available");
  }
}

mlir::Value BackEnd::generateIdentityValue(std::shared_ptr<Type> type) {
  switch (type->baseTypeEnum) {
    case BOOLEAN:
      return this->generateValue(true);
    case CHAR:
      return this->generateValue((char)0x01);
    case INTEGER:
      return this->generateValue(1);
    case REAL:
      return this->generateValue(1.0f);
    case TUPLE:
      {
        std::vector<mlir::Value> children;
        for (auto childType : type->tupleChildType) {
          children.push_back(generateIdentityValue(childType.second));
        }

        // magic code
        return this->generateValue(children);
      }
    case VECTOR:
    default:
      throw std::runtime_error("Identity not available");
  }
}

void BackEnd::deallocateObjects() {
  for (auto label : **(objectLabels.end() - 1)) {
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
    // sneaky naming trick
    mlir::LLVM::LLVMFuncOp function = builder->create<mlir::LLVM::LLVMFuncOp>(loc,"__"+signature, functionType, ::mlir::LLVM::Linkage::External, true);
    functionStack.push_back(function);

    mlir::Block *entry = function.addEntryBlock();
    builder->setInsertionPointToStart(entry);


    return currentBlock;
}

void BackEnd::generateEndFunctionDefinition(mlir::Block* returnBlock, int line) {
    builder->setInsertionPointToEnd(returnBlock);
}

void BackEnd::generateReturn(mlir::Value returnVal) {
  auto val = copyCommonType(returnVal);
  deallocateObjects();
  builder->create<mlir::LLVM::ReturnOp>(loc, val);
}

void BackEnd::generateDeclaration(std::string varName, mlir::Value value) {
  this->generateInitializeGlobalVar(varName, value);
  this->generateAssignment(varName, value);
}

/*
 * Generate an assignment to a ptr
 */
void BackEnd::generateAssignment(mlir::Value ptr, mlir::Value value) 
{
  auto assignFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("assignByReference");

  builder->create<mlir::LLVM::CallOp>(loc, 
      assignFunc, 
      mlir::ValueRange({ptr, value}));
}

void BackEnd::generateAssignment(std::string varName, mlir::Value value) {
  mlir::LLVM::GlobalOp global;

  if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(varName))) {
    llvm::errs() << "Referencing undefined variable " << varName;
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
  auto val = (*(functionStack.end()-1)).front().getArguments().vec()[index];
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
  auto currFunc = functionStack.back();
  mlir::Block *newBlock = currFunc.addBlock();
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

/*
 * Unconditionally jumps to block if `ifJump` is true
 * this is useful for ending a loop or if statement body
 *
 * This is because you cannot unconditionally jump to the endBlock
 * if there is a break/continue statement in the body, as then there might
 * be two unconditional jumps in a row
 * This breaks some MLIR rule about basic blocks so we will get an error
 *
 * also returns the new boolean value of ifJump
 */
bool BackEnd::conditionalJumpToBlock(mlir::Block *block, bool ifJump) {
    if (ifJump) {
        this->generateEnterBlock(block);
    }
    return false;
}

void BackEnd::setBuilderInsertionPoint(
    mlir::Block *block) { // set insertion point but no smart pointer:
  builder->setInsertionPointToEnd(block);
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
