#include "BackendWalker.h"

void BackendWalker::generateCode(std::shared_ptr<ASTNode> tree) {
  codeGenerator.init();
  walk(tree);
  codeGenerator.generate();
}

std::any BackendWalker::visitAssign(std::shared_ptr<AssignNode> tree) {
  auto val = std::any_cast<mlir::Value>(walk(tree->getExprNode()));

  codeGenerator.generateAssignment(tree->sym->mlirName, val);
  return 0;
}

std::any BackendWalker::visitDecl(std::shared_ptr<DeclNode> tree) {
  auto val = std::any_cast<mlir::Value>(walk(tree->getExprNode()));

  codeGenerator.generateDeclaration(tree->sym->mlirName, val);
  return 0;
}
std::any BackendWalker::visitPrint(std::shared_ptr<StreamOut> tree) {
  auto val = std::any_cast<mlir::Value>(walk(tree->getExpr()));

  if (tree->getExpr()->type->getName() == "int") {
    this->codeGenerator.print(val);
  } else {
    this->codeGenerator.printVec(val);
  }

  return 0;
}

// === EXPRESSION AST NODES ===
std::any BackendWalker::visitID(std::shared_ptr<IDNode> tree) {
  return codeGenerator.generateLoadIdentifier(tree->sym->mlirName);
}

std::any BackendWalker::visitInt(std::shared_ptr<IntNode> tree) {
  return codeGenerator.generateInteger(tree->getVal());
}

// Expr/Binary

std::any BackendWalker::visitArith(std::shared_ptr<BinaryArithNode> tree) {
  auto lhs = std::any_cast<mlir::Value>(walk(tree->getLHS()));
  auto rhs = std::any_cast<mlir::Value>(walk(tree->getRHS()));

  return codeGenerator.generateIntegerBinaryOperation(lhs, rhs, tree->op);
}

std::any BackendWalker::visitCmp(std::shared_ptr<BinaryCmpNode> tree) {
  auto lhs = std::any_cast<mlir::Value>(walk(tree->getLHS()));
  auto rhs = std::any_cast<mlir::Value>(walk(tree->getRHS()));

  return codeGenerator.generateIntegerBinaryOperation(lhs, rhs, tree->op);
}

std::any BackendWalker::visitIndex(std::shared_ptr<IndexNode> tree) {
  return 0;
}

// Expr/Vector
std::any BackendWalker::visitFilter(std::shared_ptr<FilterNode> tree) {
  mlir::Value domainVecAddr =std::any_cast<mlir::Value>(walk(tree->getVecNode()));
  mlir::Value vectorSize = codeGenerator.getVectorSize(domainVecAddr);
  mlir::Value filterVect = codeGenerator.generateVectorOfSize(vectorSize); 

  mlir::Value zero = codeGenerator.generateInteger(0);

  auto domainVarSym = tree->domainVarSym;
  codeGenerator.generateDeclaration(domainVarSym->mlirName, zero);

  mlir::Block *loopBeginBlock = codeGenerator.generateBlock();
  mlir::Block *trueBlock = codeGenerator.generateBlock();
  mlir::Block *filterTrueBlock = codeGenerator.generateBlock();
  mlir::Block *filterFalseBlock = codeGenerator.generateBlock();
  mlir::Block *exitBlock = codeGenerator.generateBlock();

  auto indexAddr = codeGenerator.generateValuePtr(zero);
  auto filterIndexAddr = codeGenerator.generateValuePtr(zero); // current size of filter. acts as INDEX for filterVec

  codeGenerator.generateEnterBlock(loopBeginBlock);
  codeGenerator.setBuilderInsertionPoint(loopBeginBlock);

  auto index = codeGenerator.generateLoadValue(indexAddr);
  auto filterIndex = codeGenerator.generateLoadValue(filterIndexAddr);
  auto comparisonResult =codeGenerator.generateIntegerBinaryOperation(index, vectorSize, LTHAN);

  codeGenerator.generateCompAndJump(trueBlock, exitBlock, comparisonResult); 
  codeGenerator.setBuilderInsertionPoint(trueBlock);

  auto currentElement = codeGenerator.generateIndexWithInteger(domainVecAddr, index);
  codeGenerator.generateAssignment(domainVarSym->mlirName, currentElement);

  mlir::Value exprResult = std::any_cast<mlir::Value>(walk(tree->getExpr()));
  auto filterComparisonResult = codeGenerator.generateIntegerBinaryOperation(exprResult, zero, NEQUAL);
  codeGenerator.generateCompAndJump(filterTrueBlock, filterFalseBlock,filterComparisonResult); 

  codeGenerator.setBuilderInsertionPoint(filterTrueBlock);

  codeGenerator.generateStoreValueInVector(filterVect, filterIndex, currentElement);
  codeGenerator.generateIncrementIndex(indexAddr);
  codeGenerator.generateIncrementIndex(filterIndexAddr);

  codeGenerator.generateEnterBlock(loopBeginBlock);

  codeGenerator.setBuilderInsertionPoint(filterFalseBlock);
  codeGenerator.generateIncrementIndex(indexAddr); 
  codeGenerator.generateEnterBlock(loopBeginBlock);

  codeGenerator.setBuilderInsertionPoint(exitBlock);

  auto filterSize = codeGenerator.generateLoadValue(filterIndexAddr);
  codeGenerator.generateSetVectorSize(filterVect, filterSize);

  return filterVect;
}

std::any BackendWalker::visitGenerator(std::shared_ptr<GeneratorNode> tree) {
  mlir::Value domainVecAddr =
      std::any_cast<mlir::Value>(walk(tree->getVecNode()));
  mlir::Value vectorSize = codeGenerator.getVectorSize(domainVecAddr);
  mlir::Value generatorVect = codeGenerator.generateVectorOfSize(vectorSize);

  mlir::Value zero = codeGenerator.generateInteger(0);

  auto domainVarSym = tree->domainVarSym;
  codeGenerator.generateDeclaration(domainVarSym->mlirName, zero);

  mlir::Block *loopBeginBlock = codeGenerator.generateBlock();
  mlir::Block *trueBlock = codeGenerator.generateBlock();
  mlir::Block *exitBlock = codeGenerator.generateBlock();

  auto indexAddr = codeGenerator.generateValuePtr(zero);
  codeGenerator.generateEnterBlock(loopBeginBlock);
  codeGenerator.setBuilderInsertionPoint(loopBeginBlock);

  auto index = codeGenerator.generateLoadValue(indexAddr);
  auto comparisonResult =
      codeGenerator.generateIntegerBinaryOperation(index, vectorSize, LTHAN);

  codeGenerator.generateCompAndJump(trueBlock, exitBlock, comparisonResult);
  codeGenerator.setBuilderInsertionPoint(trueBlock);

  auto domainVar = codeGenerator.generateIndexWithInteger(domainVecAddr, index);
  codeGenerator.generateAssignment(domainVarSym->mlirName, domainVar);

  mlir::Value exprResult = std::any_cast<mlir::Value>(walk(tree->getExpr()));

  codeGenerator.generateStoreValueInVector(generatorVect, index, exprResult);
  codeGenerator.generateIncrementIndex(indexAddr);

  codeGenerator.generateEnterBlock(loopBeginBlock);
  codeGenerator.setBuilderInsertionPoint(exitBlock);

  return generatorVect;
}

std::any BackendWalker::visitRangeVec(std::shared_ptr<RangeVecNode> tree) {
  auto lower = std::any_cast<mlir::Value>(walk(tree->getStart()));
  auto upper = std::any_cast<mlir::Value>(walk(tree->getEnd()));

  return this->codeGenerator.generateVectorFromRange(lower, upper);
}

// === BLOCK AST NODES ===

std::any BackendWalker::visitConditional(std::shared_ptr<ConditionalNode> tree) {
  mlir::Block *trueBlock = codeGenerator.generateBlock();
  mlir::Block *falseBlock = codeGenerator.generateBlock();
  mlir::Value result = std::any_cast<mlir::Value>(walk(tree->condition));

  codeGenerator.generateCompAndJump(trueBlock, falseBlock, result);
  codeGenerator.setBuilderInsertionPoint(trueBlock);

  walkChildren(tree);

  codeGenerator.generateEnterBlock(falseBlock);
  codeGenerator.setBuilderInsertionPoint(falseBlock);

  return 0;
}

std::any BackendWalker::visitLoop(std::shared_ptr<LoopNode> tree) {
  mlir::Block *loopBeginBlock = codeGenerator.generateBlock();
  mlir::Block *trueBlock = codeGenerator.generateBlock();
  mlir::Block *falseBlock = codeGenerator.generateBlock();

  codeGenerator.generateEnterBlock(loopBeginBlock);
  codeGenerator.setBuilderInsertionPoint(loopBeginBlock);

  mlir::Value exprResult = std::any_cast<mlir::Value>(walk(tree->condition));
  codeGenerator.generateCompAndJump(trueBlock, falseBlock, exprResult);

  codeGenerator.setBuilderInsertionPoint(trueBlock);
  walkChildren(tree);

  codeGenerator.generateEnterBlock(loopBeginBlock);
  codeGenerator.setBuilderInsertionPoint(falseBlock);
  return 0;
}

