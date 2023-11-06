#include "BackendWalker.h"
#include "mlir/IR/Value.h"
#include <stdexcept>

void BackendWalker::generateCode(std::shared_ptr<ASTNode> tree) {
  codeGenerator.init();
  walkChildren(tree);
  codeGenerator.deallocateObjects();
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

  this->codeGenerator.printCommonType(val);

  return 0;
}

// === EXPRESSION AST NODES ===
std::any BackendWalker::visitID(std::shared_ptr<IDNode> tree) {
  return codeGenerator.generateLoadIdentifier(tree->sym->mlirName);
}

std::any BackendWalker::visitInt(std::shared_ptr<IntNode> tree) {
  return codeGenerator.generateValue(tree->getVal());
}

std::any BackendWalker::visitReal(std::shared_ptr<RealNode> tree) {
  return codeGenerator.generateValue(tree->getVal());
}

std::any BackendWalker::visitChar(std::shared_ptr<CharNode> tree) {
  return codeGenerator.generateValue(tree->getVal());
}

std::any BackendWalker::visitBool(std::shared_ptr<BoolNode> tree) {
  return codeGenerator.generateValue(tree->getVal());
}

std::any BackendWalker::visitTuple(std::shared_ptr<TupleNode> tree) {
  std::vector<mlir::Value> values;

  for (auto node : tree->getVal()) {
    values.push_back(std::any_cast<mlir::Value>(walk(node)));
  }

  return codeGenerator.generateValue(values);
}

// Expr/Binary

std::any BackendWalker::visitArith(std::shared_ptr<BinaryArithNode> tree) {
  auto lhs = std::any_cast<mlir::Value>(walk(tree->getLHS()));
  auto rhs = std::any_cast<mlir::Value>(walk(tree->getRHS()));

  return codeGenerator.performBINOP(lhs, rhs, tree->op);
}

std::any BackendWalker::visitCmp(std::shared_ptr<BinaryCmpNode> tree) {
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

  // walk all the conditions, then we have a conditional jump chain.
  for (auto condition : tree->conditions) {
    walk(condition);
  }

  //codeGenerator.generateCompAndJump(trueBlock, falseBlock, result);
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

std::any BackendWalker::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
  std::cout << "here " << std::endl;
  // differnece in how these are called. functions get references, procedures
  // values? Or the other way around w/e
  
  if (tree->body) {
    // for now we don't proper return values
    auto block = codeGenerator.generateFunctionDefinition(tree->nameSym->name, 
        tree->orderedArgs.size(), 
        true);
    walk(tree->body);
    mlir::Value val; // void
    codeGenerator.generateEndFunctionDefinition(block, val);
  }

  return walkChildren(tree);
}

std::any BackendWalker::visitFunction(std::shared_ptr<FunctionNode> tree) {
  if (tree->body) {
    // for now we don't proper return values
    auto block = codeGenerator.generateFunctionDefinition(tree->funcNameSym->name, 
        tree->orderedArgs.size(), 
        true);
    walk(tree->body);
    mlir::Value val; // void
    codeGenerator.generateEndFunctionDefinition(block, val);
  }
  return walkChildren(tree);
}

std::any BackendWalker::visitBlock(std::shared_ptr<BlockNode> tree) {
  // deallocate here later.
  return walkChildren(tree);
}


