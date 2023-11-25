#include "BackendWalker.h"
#include "ASTNode/Expr/CastNode.h"
#include "ASTWalker.h"
#include "Operands/BINOP.h"
#include "Types/TYPES.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/IR/Value.h"
#include <memory>
#include <stdexcept>
//#define DEBUG


std::any BackendWalker::walk(std::shared_ptr<ASTNode> tree) {
  // stop reading code if theres a return won't be reached
  if (!this->returnDropped) {
    return gazprea::ASTWalker::walk(tree);
  }
  return 0;
}

void BackendWalker::generateCode(std::shared_ptr<ASTNode> tree) {
#ifdef DEBUG
  std::cout << "CODE GENERATION\n";
  std::cout << INTEGER << REAL << std::endl;
#endif 

  codeGenerator.init();
  walkChildren(tree);
  //codeGenerator.deallocateObjects();
  codeGenerator.generate();
}

std::any BackendWalker::visitAssign(std::shared_ptr<AssignNode> tree) {
  auto val = std::any_cast<mlir::Value>(walk(tree->getRvalue()));
  auto exprList = std::dynamic_pointer_cast<ExprListNode>(tree->getLvalue());
 
  if (exprList->children.size() == 1) {
    auto dest = std::any_cast<mlir::Value>(walk(exprList->children[0]));
    codeGenerator.generateAssignment(dest, val);
  } else {
    for (int i = 0 ; i < exprList->children.size() ; i++) {
      auto dest = std::any_cast<mlir::Value>(walk(exprList->children[i]));
      auto indexedValue = codeGenerator.indexCommonType(val, codeGenerator.generateValue(i));

      auto castedIndexedVal = codeGenerator.possiblyCast(indexedValue, tree->evaluatedType);
      codeGenerator.generateAssignment(dest, castedIndexedVal);
    }
  }

  return 0;
}

std::any BackendWalker::visitDecl(std::shared_ptr<DeclNode> tree) {
  auto val = std::any_cast<mlir::Value>(walk(tree->getExprNode()));

  auto castedVal = codeGenerator.possiblyCast(val, tree->evaluatedType);

  codeGenerator.generateDeclaration(tree->sym->mlirName, castedVal);
  return 0;
}

std::any BackendWalker::visitStreamOut(std::shared_ptr<StreamOut> tree) {
  auto val = std::any_cast<mlir::Value>(walk(tree->getExpr()));

  this->codeGenerator.streamOut(val);

  return 0;
}

std::any BackendWalker::visitStreamIn(std::shared_ptr<StreamIn> tree) {
    auto val = std::any_cast<mlir::Value>(walk(tree->getExpr()));

    this->codeGenerator.streamIn(val);

    return 0;
}

// === EXPRESSION AST NODES ===
std::any BackendWalker::visitID(std::shared_ptr<IDNode> tree) {
  // might be arg
  if (tree->sym->index >= 0) {
    return codeGenerator.generateLoadArgument(tree->sym->index);
  } else {
    std::cout << "trying to load " << tree->sym->mlirName << std::endl;
    return codeGenerator.generateLoadIdentifier(tree->sym->mlirName);
  }
}

std::any BackendWalker::visitIdentity(std::shared_ptr<IdentityNode> tree) {

    return codeGenerator.generateIdentityValue(tree->evaluatedType->baseTypeEnum);
}

std::any BackendWalker::visitNull(std::shared_ptr<NullNode> tree) {
    return codeGenerator.generateNullValue(tree->evaluatedType->baseTypeEnum);
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

std::any BackendWalker::visitVector(std::shared_ptr<VectorNode> tree) {
  std::vector<mlir::Value> values;

  for (auto node : tree->getElements()) {
    values.push_back(std::any_cast<mlir::Value>(walk(node)));
  }

  return codeGenerator.generateValue(values);
}

std::any BackendWalker::visitTuple(std::shared_ptr<TupleNode> tree) {
  std::vector<mlir::Value> values;

  for (auto node : tree->getVal()) {
    values.push_back(std::any_cast<mlir::Value>(walk(node)));
  }

  return codeGenerator.generateValue(values);
}

std::any BackendWalker::visitTupleIndex(std::shared_ptr<TupleIndexNode> tree) {
  mlir::Value indexee;

  // indexee isn't an expression. HACK
  if (tree->sym->index >= 0) {
    indexee = codeGenerator.generateLoadArgument(tree->sym->index);
  } else {
    indexee =codeGenerator.generateLoadIdentifier(tree->sym->mlirName);
  }

  return codeGenerator.indexCommonType(indexee, codeGenerator.generateValue(tree->index));
}

// Expr/Binary
std::any BackendWalker::visitCast(std::shared_ptr<CastNode> tree) {
  auto val = std::any_cast<mlir::Value>(walk(tree->getExpr()));
  auto type = tree->evaluatedType->baseTypeEnum;

  return codeGenerator.cast(val, type);
}

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
  auto indexee = std::any_cast<mlir::Value>(walk(tree->getIndexee()));

  // im sad we don't use recursion more
  for (int i = 1 ; i < tree->children.size() ; i ++) {
    auto indexor = std::any_cast<mlir::Value>(walk(tree->children[i]));
    indexee = codeGenerator.indexCommonType(indexee, indexor);
  }

  return indexee;
}

// Expr/Vector
std::any BackendWalker::visitFilter(std::shared_ptr<FilterNode> tree) {
  // what we are filtering from
  auto filteree = std::any_cast<mlir::Value>(walk(tree->getDomain()));
  auto one = codeGenerator.generateValue(1);

  // max amount of filters
  auto maxFiltered = codeGenerator.generateValue((int)tree->getExprList().size());

  // empty filter we are appending to
  auto filter = codeGenerator.generateValue(maxFiltered);
  std::vector<mlir::Value> argument;
  argument.push_back(filteree);
  auto domain = codeGenerator.generateValue(0);
  codeGenerator.generateDeclaration(tree->domainVarSym->mlirName, domain); 

  auto maxVectorSize = codeGenerator.generateCallNamed("length", argument);

  for (int i = 0 ; i < tree->getExprList().size() ; i++) {
    auto newVector = codeGenerator.generateValue(maxVectorSize);

    mlir::Block *loopBeginBlock = codeGenerator.generateBlock();
    mlir::Block *trueBlock = codeGenerator.generateBlock();
    mlir::Block *exitBlock = codeGenerator.generateBlock();
    auto index = codeGenerator.generateValue(0);

    codeGenerator.generateEnterBlock(loopBeginBlock);
    codeGenerator.setBuilderInsertionPoint(loopBeginBlock);
 
    auto inBounds = codeGenerator.performBINOP(index, maxVectorSize, LTHAN);

    codeGenerator.generateCompAndJump(trueBlock, exitBlock, codeGenerator.downcastToBool(inBounds)); 

    codeGenerator.setBuilderInsertionPoint(trueBlock);
    auto indexedVal = codeGenerator.indexCommonType(filteree, index);
    codeGenerator.generateAssignment(domain, indexedVal);

    auto result = std::any_cast<mlir::Value>(walk(tree->getExprList()[i]));


    mlir::Block *trueResult = codeGenerator.generateBlock();
    mlir::Block *falseResult= codeGenerator.generateBlock();
    codeGenerator.generateCompAndJump(trueResult, falseResult, codeGenerator.downcastToBool(result)) ;
    
    codeGenerator.setBuilderInsertionPoint(trueResult);

    codeGenerator.appendCommon(newVector, indexedVal);

    codeGenerator.generateEnterBlock(falseResult);
    codeGenerator.setBuilderInsertionPoint(falseResult);

    codeGenerator.generateAssignment(index, codeGenerator.performBINOP(index, one, ADD));

    codeGenerator.generateEnterBlock(loopBeginBlock);
    codeGenerator.setBuilderInsertionPoint(exitBlock);
    codeGenerator.appendCommon(filter, newVector);
  }

  return filter;
}

std::any BackendWalker::visitGenerator(std::shared_ptr<GeneratorNode> tree) {

  if (tree->getVectDomain()) {
    auto baseVec = std::any_cast<mlir::Value>(walk(tree->getVectDomain()));

    // we do a little indexing
    auto index = codeGenerator.generateValue(0);
    auto domain = codeGenerator.generateValue(0);
    auto one = codeGenerator.generateValue(1);

    // build arg list
    std::vector<mlir::Value> argument;
    argument.push_back(baseVec);

    codeGenerator.generateDeclaration(tree->domainVar1Sym->mlirName, domain);

    auto length = codeGenerator.generateCallNamed("length", argument);
    auto generatorVector = codeGenerator.generateValue(length);

    mlir::Block *loopBeginBlock = codeGenerator.generateBlock();
    mlir::Block *trueBlock = codeGenerator.generateBlock();
    mlir::Block *exitBlock = codeGenerator.generateBlock();

    codeGenerator.generateEnterBlock(loopBeginBlock);
    codeGenerator.setBuilderInsertionPoint(loopBeginBlock);
 
    auto inBounds = codeGenerator.performBINOP(index, length, LTHAN);

    codeGenerator.generateCompAndJump(trueBlock, exitBlock, codeGenerator.downcastToBool(inBounds)); 

    codeGenerator.setBuilderInsertionPoint(trueBlock);
    auto indexedVal = codeGenerator.indexCommonType(baseVec, index);
    codeGenerator.generateAssignment(domain, indexedVal);

    auto result = std::any_cast<mlir::Value>(walk(tree->getExpr()));

    codeGenerator.appendCommon(generatorVector, result);
    codeGenerator.generateAssignment(index, codeGenerator.performBINOP(index, one, ADD));

    codeGenerator.generateEnterBlock(loopBeginBlock);
    codeGenerator.setBuilderInsertionPoint(exitBlock);

    return generatorVector;
  } else {
    auto row = std::any_cast<mlir::Value>(walk(tree->getMatrixDomain().first));
    auto column = std::any_cast<mlir::Value>(walk(tree->getMatrixDomain().second));

    // we do a little indexing
    auto rowIndex = codeGenerator.generateValue(0);
    auto rowDomain = codeGenerator.generateValue(0);
    auto colDomain = codeGenerator.generateValue(0);

    auto one = codeGenerator.generateValue(1);

    // build arg list
    std::vector<mlir::Value> rowArgument;
    std::vector<mlir::Value> colArgument;

    codeGenerator.generateDeclaration(tree->domainVar1Sym->mlirName, rowDomain);
    codeGenerator.generateDeclaration(tree->domainVar2Sym->mlirName, colDomain);

    rowArgument.push_back(row);
    colArgument.push_back(column);

    auto rowLength = codeGenerator.generateCallNamed("length", rowArgument);
    auto colLength = codeGenerator.generateCallNamed("length", colArgument);

    auto generatorVector = codeGenerator.generateValue(rowLength);

    mlir::Block *matrixBeginBlock= codeGenerator.generateBlock();
    mlir::Block *matrixTrueBlock= codeGenerator.generateBlock();
    mlir::Block *matrixExitBlock= codeGenerator.generateBlock();

    codeGenerator.generateEnterBlock(matrixBeginBlock);
    codeGenerator.setBuilderInsertionPoint(matrixBeginBlock);
    auto inBoundsRow = codeGenerator.performBINOP(rowIndex, rowLength, LTHAN);
    codeGenerator.generateCompAndJump(matrixTrueBlock, matrixExitBlock, codeGenerator.downcastToBool(inBoundsRow)); 
    codeGenerator.setBuilderInsertionPoint(matrixTrueBlock);

    codeGenerator.appendCommon(generatorVector, codeGenerator.generateValue(colLength));
    codeGenerator.generateAssignment(rowIndex, codeGenerator.performBINOP(rowIndex,one , ADD));

    codeGenerator.generateEnterBlock(matrixBeginBlock);
    codeGenerator.setBuilderInsertionPoint(matrixExitBlock);

    rowIndex = codeGenerator.generateValue(0);

    mlir::Block *rowBeginBlock= codeGenerator.generateBlock();
    mlir::Block *rowTrueBlock= codeGenerator.generateBlock();
    mlir::Block *rowExitBlock= codeGenerator.generateBlock();

    codeGenerator.generateEnterBlock(rowBeginBlock);
    codeGenerator.setBuilderInsertionPoint(rowBeginBlock);
 
    inBoundsRow = codeGenerator.performBINOP(rowIndex, rowLength, LTHAN);
    codeGenerator.generateCompAndJump(rowTrueBlock, rowExitBlock, codeGenerator.downcastToBool(inBoundsRow)); 
    codeGenerator.setBuilderInsertionPoint(rowTrueBlock);
    auto colIndex = codeGenerator.generateValue(0);

    /* COL ========================= */
      mlir::Block *colBeginBlock= codeGenerator.generateBlock();
      mlir::Block *colTrueBlock= codeGenerator.generateBlock();
      mlir::Block *colExitBlock= codeGenerator.generateBlock();

      codeGenerator.generateEnterBlock(colBeginBlock);
      codeGenerator.setBuilderInsertionPoint(colBeginBlock);

      auto inBoundsCol = codeGenerator.performBINOP(colIndex, colLength, LTHAN);
      codeGenerator.generateCompAndJump(colTrueBlock, colExitBlock, codeGenerator.downcastToBool(inBoundsCol)); 

      codeGenerator.setBuilderInsertionPoint(colTrueBlock);

      auto indexedRow = codeGenerator.indexCommonType(row, rowIndex);
      auto indexedCol = codeGenerator.indexCommonType(column, colIndex);
    
      codeGenerator.generateAssignment(rowDomain, indexedRow);
      codeGenerator.generateAssignment(colDomain, indexedCol);

      auto result = std::any_cast<mlir::Value>(walk(tree->getExpr()));

      codeGenerator.appendCommon(codeGenerator.indexCommonType(generatorVector, rowIndex), result);

      codeGenerator.generateAssignment(colIndex, codeGenerator.performBINOP(colIndex, one, ADD));

      codeGenerator.generateEnterBlock(colBeginBlock);
      codeGenerator.setBuilderInsertionPoint(colExitBlock);
    /* COL ========================= */

    codeGenerator.generateAssignment(rowIndex, codeGenerator.performBINOP(rowIndex, one, ADD));

    codeGenerator.generateEnterBlock(rowBeginBlock);
    codeGenerator.setBuilderInsertionPoint(rowExitBlock);

    return generatorVector;
  }
}

std::any BackendWalker::visitRangeVec(std::shared_ptr<RangeVecNode> tree) {
  auto lower = std::any_cast<mlir::Value>(walk(tree->getStart()));
  auto upper = std::any_cast<mlir::Value>(walk(tree->getEnd()));

  return codeGenerator.generateValue(lower, upper);
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

    // return was dropped during walk, don't need to bound back
    if (!this->returnDropped) {
      codeGenerator.conditionalJumpToBlock(endBlock, !earlyReturn);
    }

    this->returnDropped = false;
    this->earlyReturn = false;

    codeGenerator.setBuilderInsertionPoint(falseBlocks[i]);
  }

  // if there is an "else" clause, we will have one more "body" node
  if (tree->bodies.size() > tree->conditions.size()) {
    walk(tree->bodies[tree->bodies.size() - 1]);
  }

  codeGenerator.conditionalJumpToBlock(endBlock, !earlyReturn);
  this->earlyReturn = false;
  this->returnDropped = false;
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

std::any BackendWalker::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
  if (tree->body) {
    // for now we don't proper return values, assume everything void
    auto block = codeGenerator.generateFunctionDefinition(tree->nameSym->name,
        tree->orderedArgs.size(),
        false);
    walk(tree->body);
    codeGenerator.generateEndFunctionDefinition(block, tree->loc());
    codeGenerator.verifyFunction(tree->loc(), "Procedure " + tree->nameSym->name);
    this->returnDropped = false;
  }

  return 0;
}

std::any BackendWalker::visitFunction(std::shared_ptr<FunctionNode> tree) {
  if (tree->body) {

    auto block = codeGenerator.generateFunctionDefinition(tree->funcNameSym->name,
        tree->orderedArgs.size(),
        false);
    walk(tree->body);

    codeGenerator.generateEndFunctionDefinition(block, tree->loc());
    codeGenerator.verifyFunction(tree->loc(), "Function " + tree->funcNameSym->name);
    this->returnDropped = false;
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
  this->returnDropped = true;
  return 0;
}

std::any BackendWalker::visitBlock(std::shared_ptr<BlockNode> tree) {
  codeGenerator.pushScope();

  auto returnVal = walkChildren(tree);

  codeGenerator.popScope();
  return returnVal;
}
