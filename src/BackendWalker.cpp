#include "BackendWalker.h"
#include <stdexcept>

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
std::any BackendWalker::visitPrint(std::shared_ptr<PrintNode> tree) {
  auto val = std::any_cast<mlir::Value>(walk(tree->getExpr()));

  this->codeGenerator.printCommonType(val);

  return 0;
}

// === EXPRESSION AST NODES ===
std::any BackendWalker::visitID(std::shared_ptr<IDNode> tree) {
  return codeGenerator.generateLoadIdentifier(tree->sym->mlirName);
}

std::any BackendWalker::visitInt(std::shared_ptr<IntNode> tree) {
  auto result = codeGenerator.generateValue(tree->getVal());
  return result;
}

// Expr/Binary
std::any BackendWalker::visitArith(std::shared_ptr<ArithNode> tree) {
  auto lhs = std::any_cast<mlir::Value>(walk(tree->getLHS()));
  auto rhs = std::any_cast<mlir::Value>(walk(tree->getRHS()));

  return codeGenerator.performBINOP(lhs, rhs, tree->op);
}

std::any BackendWalker::visitCmp(std::shared_ptr<CmpNode> tree) {
  auto lhs = std::any_cast<mlir::Value>(walk(tree->getLHS()));
  auto rhs = std::any_cast<mlir::Value>(walk(tree->getRHS()));

  return codeGenerator.performBINOP(lhs, rhs, tree->op);
}

std::any BackendWalker::visitIndex(std::shared_ptr<IndexNode> tree) {
  return 0;
}

// Expr/Vector
std::any BackendWalker::visitFilter(std::shared_ptr<FilterNode> tree) {
  throw std::runtime_error("Not implemented!");
  return walkChildren(tree);
}

std::any BackendWalker::visitGenerator(std::shared_ptr<GeneratorNode> tree) {
  throw std::runtime_error("Not implemented!");
  return walkChildren(tree);
}

std::any BackendWalker::visitRangeVec(std::shared_ptr<RangeVecNode> tree) {
  throw std::runtime_error("Not implemented!");
  return walkChildren(tree);
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

