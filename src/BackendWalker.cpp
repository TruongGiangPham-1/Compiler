#include "BackendWalker.h"
#include <stdexcept>

void BackendWalker::generateCode(std::shared_ptr<ASTNode> tree) {
  codeGenerator.init();
  walk(tree);
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
  return codeGenerator.generateLoadIdentifier(tree->sym->mlirName);
}

std::any BackendWalker::visitInt(std::shared_ptr<IntNode> tree) {
  auto result = codeGenerator.generateValue(tree->getVal());
  return result;
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
    codeGenerator.conditionalJumpToBlock(endBlock, !earlyReturn);
    this->earlyReturn = false;

    codeGenerator.setBuilderInsertionPoint(falseBlocks[i]);
  }

  // if there is an "else" clause, we will have one more "body" node
  if (tree->bodies.size() > tree->conditions.size()) {
    walk(tree->bodies[tree->bodies.size() - 1]);
  }

  codeGenerator.conditionalJumpToBlock(endBlock, !earlyReturn);
  this->earlyReturn = false;
  codeGenerator.setBuilderInsertionPoint(endBlock);

  return 0;
}

std::any BackendWalker::visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree) {
  auto loopBody = codeGenerator.generateBlock(); // start of loop
  auto loopExit = codeGenerator.generateBlock(); // the rest of the program

  this->loopBlocks.push_back(std::make_pair(loopBody, loopExit));

  // body of loop
  codeGenerator.generateEnterBlock(loopBody);
  codeGenerator.setBuilderInsertionPoint(loopBody);
  walk(tree->getBody());
  codeGenerator.conditionalJumpToBlock(loopBody, !earlyReturn);
  this->earlyReturn = false;

  // loop exit
  codeGenerator.setBuilderInsertionPoint(loopExit);
  this->loopBlocks.pop_back();

  return 0;
}

std::any BackendWalker::visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree) {
  auto loopCheck = codeGenerator.generateBlock(); // check
  auto loopBody= codeGenerator.generateBlock(); // body
  auto loopExit = codeGenerator.generateBlock(); // the rest of the program

  this->loopBlocks.push_back(std::make_pair(loopCheck, loopExit));

  // check conditional
  codeGenerator.generateEnterBlock(loopCheck);
  codeGenerator.setBuilderInsertionPoint(loopCheck);
  auto condResult = std::any_cast<mlir::Value>(walk(tree->getCondition()));
  auto condBool = codeGenerator.downcastToBool(condResult);
  codeGenerator.generateCompAndJump(loopBody, loopExit, condBool);

  // body of loop
  codeGenerator.setBuilderInsertionPoint(loopBody);
  walk(tree->getBody());
  codeGenerator.conditionalJumpToBlock(loopCheck, !earlyReturn);
  this->earlyReturn = false;

  // loop exit
  codeGenerator.setBuilderInsertionPoint(loopExit);
  this->loopBlocks.pop_back();

  return 0;
}

std::any BackendWalker::visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree) {
  // check goes after the body
  auto loopBody = codeGenerator.generateBlock();
  auto loopCheck = codeGenerator.generateBlock();
  auto loopExit = codeGenerator.generateBlock();

  this->loopBlocks.push_back(std::make_pair(loopBody, loopExit));

  // body of loop
  codeGenerator.generateEnterBlock(loopBody);
  codeGenerator.setBuilderInsertionPoint(loopBody);
  walk(tree->getBody());
  codeGenerator.conditionalJumpToBlock(loopCheck, !earlyReturn);
  this->earlyReturn = false;

  // conditional
  codeGenerator.setBuilderInsertionPoint(loopCheck);
  auto condResult = std::any_cast<mlir::Value>(walk(tree->getCondition()));
  auto condBool = codeGenerator.downcastToBool(condResult);
  codeGenerator.generateCompAndJump(loopBody, loopExit, condBool);

  // loop exit
  codeGenerator.setBuilderInsertionPoint(loopExit);
  this->loopBlocks.pop_back();

  return 0;
}

std::any BackendWalker::visitBreak(std::shared_ptr<BreakNode> tree) {
    if (this->loopBlocks.empty()) {
        throw StatementError(tree->loc(), "Break statement outside of loop");
    }

    auto loopExit = this->loopBlocks.back().second;
    codeGenerator.generateEnterBlock(loopExit);
    this->earlyReturn = true;

    return 0;
}

std::any BackendWalker::visitContinue(std::shared_ptr<ContinueNode> tree) {
    if (this->loopBlocks.empty()) {
        throw StatementError(tree->loc(), "Continue statement outside of loop");
    }

    auto loopBody = this->loopBlocks.back().first;
    codeGenerator.generateEnterBlock(loopBody);
    this->earlyReturn = true;

    return 0;
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
