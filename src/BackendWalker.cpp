#include "BackendWalker.h"
#include "mlir/IR/Value.h"
#include <stdexcept>

void BackendWalker::generateCode(std::shared_ptr<ASTNode> tree) {
  codeGenerator.init();
  walkChildren(tree);
  //codeGenerator.deallocateObjects();
  codeGenerator.generate();
}

std::any BackendWalker::visitAssign(std::shared_ptr<AssignNode> tree) {
  auto val = std::any_cast<mlir::Value>(walk(tree->getRvalue()));

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
  // might be arg
  if (tree->sym->index >= 0) {
    return codeGenerator.generateLoadArgument(tree->sym->index);
  } else {
    return codeGenerator.generateLoadIdentifier(tree->sym->mlirName);
  }
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

std::any BackendWalker::visitUnaryArith(std::shared_ptr<UnaryArithNode> tree) {
  auto expr = std::any_cast<mlir::Value>(walk(tree->getExpr()));
  return codeGenerator.performUNARYOP(expr, tree->op);
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
  // create all our block ahead of time
  // for each condition, we need a true and a false mlir::Block.
  // these go at the beginning and end of each if statement's body, respectively
  // finally, we need an end mlir::Block placed at the end of all conditions
  std::vector<mlir::Block *> trueBlocks;
  std::vector<mlir::Block *> falseBlocks;
  for (int i = 0; i < tree->conditions.size(); i++) {
    trueBlocks.push_back(codeGenerator.generateBlock());
    falseBlocks.push_back(codeGenerator.generateBlock());
  }
  mlir::Block *endBlock = codeGenerator.generateBlock();

  // now, go through the conditions and bodies and generate the code
  // the number of bodies is never less than the number of conditions
  for (int i = 0; i < tree->conditions.size(); i++) {
    auto condResultVal = std::any_cast<mlir::Value>(walk(tree->conditions[i]));
    auto condResultBool = codeGenerator.downcastToBool(condResultVal);

    codeGenerator.generateCompAndJump(trueBlocks[i], falseBlocks[i], condResultBool);

    codeGenerator.setBuilderInsertionPoint(trueBlocks[i]);
    walk(tree->bodies[i]);

    codeGenerator.generateEnterBlock(endBlock);
    codeGenerator.setBuilderInsertionPoint(falseBlocks[i]);
  }

  // if there is an "else" clause, we will have one more "body" node
  if (tree->bodies.size() > tree->conditions.size()) {
    walk(tree->bodies[tree->bodies.size() - 1]);
    codeGenerator.generateEnterBlock(endBlock);
  }

  codeGenerator.setBuilderInsertionPoint(endBlock);

  return 0;
}

std::any BackendWalker::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
  if (tree->body) {
    // for now we don't proper return values, assume everything void
    auto block = codeGenerator.generateFunctionDefinition(tree->nameSym->name, 
        tree->orderedArgs.size(), 
        false);
    walk(tree->body);
    codeGenerator.generateEndFunctionDefinition(block);
  }

  return 0;
}

std::any BackendWalker::visitFunction(std::shared_ptr<FunctionNode> tree) {
  if (tree->body) {

    auto block = codeGenerator.generateFunctionDefinition(tree->funcNameSym->name, 
        tree->orderedArgs.size(), 
        false);
    walk(tree->body);


    codeGenerator.generateEndFunctionDefinition(block);
  }
  return 0;
}

std::any BackendWalker::visitCall(std::shared_ptr<CallNode> tree) {
  std::vector<mlir::Value> arguments;

  for (auto argument : tree->children) {
    arguments.push_back(std::any_cast<mlir::Value>(walk(argument)));
  }

  auto result = codeGenerator.generateCallNamed(tree->CallName->name, arguments);
  return result;
}

std::any BackendWalker::visitReturn(std::shared_ptr<ReturnNode> tree) {
  codeGenerator.generateReturn(std::any_cast<mlir::Value>(walk(tree->returnExpr)));
  return 0;
}

std::any BackendWalker::visitBlock(std::shared_ptr<BlockNode> tree) {
  // TODO deallocate here later.
  return walkChildren(tree);
}

//std::any BackendWalker::visitLoop(std::shared_ptr<LoopNode> tree) {
//  mlir::Block *loopBeginBlock = codeGenerator.generateBlock();
//  mlir::Block *trueBlock = codeGenerator.generateBlock();
//  mlir::Block *falseBlock = codeGenerator.generateBlock();
//
//  codeGenerator.generateEnterBlock(loopBeginBlock);
//  codeGenerator.setBuilderInsertionPoint(loopBeginBlock);
//
//  mlir::Value exprResult = std::any_cast<mlir::Value>(walk(tree->getCondition()));
//  codeGenerator.generateCompAndJump(trueBlock, falseBlock, exprResult);
//
//  codeGenerator.setBuilderInsertionPoint(trueBlock);
//  walkChildren(tree);
//
//  codeGenerator.generateEnterBlock(loopBeginBlock);
//  codeGenerator.setBuilderInsertionPoint(falseBlock);
//  return 0;
//}
//
