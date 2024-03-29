#include "BackendWalker.h"
#include "ASTNode/Expr/CastNode.h"
#include "ASTNode/Type/MatrixTypeNode.h"
#include "ASTNode/Type/TupleTypeNode.h"
#include "ASTNode/Type/VectorTypeNode.h"
#include "ASTWalker.h"
#include "Operands/BINOP.h"
#include "Types/TYPES.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/IR/Value.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <strings.h>
// #define DEBUG

std::any BackendWalker::walk(std::shared_ptr<ASTNode> tree)
{
    // stop reading code if theres a return won't be reached
    if (!this->returnDropped) {
        return gazprea::ASTWalker::walk(tree);
    }
    return 0;
}

void BackendWalker::generateCode(std::shared_ptr<ASTNode> tree)
{
#ifdef DEBUG
    std::cout << "CODE GENERATION\n";
    std::cout << INTEGER << REAL << std::endl;
#endif

    codeGenerator.init();
    //  codeGenerator.functionShowcase();
    walkChildren(tree);
    // codeGenerator.deallocateObjects();
    codeGenerator.generate();
}

std::any BackendWalker::visitAssign(std::shared_ptr<AssignNode> tree)
{
    auto val = std::any_cast<mlir::Value>(walk(tree->getRvalue()));
    auto exprList = std::dynamic_pointer_cast<ExprListNode>(tree->getLvalue());

    if (exprList->children.size() == 1) {
        this->fetchRaw = true;
        auto dest = std::any_cast<mlir::Value>(walk(exprList->children[0]));
        this->fetchRaw = false;
        codeGenerator.generateAssignment(dest, val);
    } else {
        for (int i = 0; i < exprList->children.size(); i++) {
            this->fetchRaw = true;
            auto dest = std::any_cast<mlir::Value>(walk(exprList->children[i]));
            this->fetchRaw = false;
            auto indexedValue = codeGenerator.indexCommonType(val, codeGenerator.generateValue(i + 1));

            codeGenerator.generateAssignment(dest, indexedValue);
        }
    }

    return 0;
}

std::any BackendWalker::visitDecl(std::shared_ptr<DeclNode> tree)
{
    mlir::Value initializedType;
    // dynamic typecheck if lhs type exists, otherwise assign

    auto val = std::any_cast<mlir::Value>(walk(tree->getExprNode()));

    this->inferenceContext.push_back(val);

    if (tree->getTypeNode()) {
        initializedType = std::any_cast<mlir::Value>(walk(tree->getTypeNode()));
        codeGenerator.generateAssignment(initializedType, val);
    } else {
        initializedType = val;
    }

    this->inferenceContext.pop_back();

    if (tree->sym->functionStackIndex >= 0) {
        codeGenerator.generateAssignment(
            codeGenerator.indexCommonType(
                this->variableStack.back(),
                codeGenerator.generateValue(tree->sym->declarationIndex)),
            initializedType);
    } else {
        codeGenerator.generateDeclaration(tree->sym->mlirName, initializedType);
    }

    return 0;
}

std::any BackendWalker::visitType(std::shared_ptr<TypeNode> tree)
{
    if (!tree->evaluatedType->vectorInnerTypes.empty() && tree->evaluatedType->vectorOrMatrixEnum == VECTOR && tree->evaluatedType->vectorInnerTypes[0]->vectorOrMatrixEnum != VECTOR) {
        auto mtree = std::dynamic_pointer_cast<VectorTypeNode>(tree);

        mlir::Value size;
        if (mtree->getSize()) {
            size = std::any_cast<mlir::Value>(walk(mtree->getSize()));
        } else {
            std::vector<mlir::Value> arguments;
            arguments.push_back(*(this->inferenceContext.end() - 1));
            size = codeGenerator.generateCallNamed("length", arguments);
        }

        auto one = codeGenerator.generateValue(1);

        auto newVector = codeGenerator.generateValue(size, tree->isString);

        mlir::Block* loopBeginBlock = codeGenerator.generateBlock();
        mlir::Block* trueBlock = codeGenerator.generateBlock();
        mlir::Block* exitBlock = codeGenerator.generateBlock();

        auto index = codeGenerator.generateValue(0);

        codeGenerator.generateEnterBlock(loopBeginBlock);
        codeGenerator.setBuilderInsertionPoint(loopBeginBlock);

        auto inBounds = codeGenerator.performBINOP(index, size, LTHAN);

        codeGenerator.generateCompAndJump(trueBlock, exitBlock, codeGenerator.downcastToBool(inBounds));

        codeGenerator.setBuilderInsertionPoint(trueBlock);

        auto result = codeGenerator.generateNullValue(mtree->evaluatedType);

        codeGenerator.appendCommon(newVector, result);

        codeGenerator.generateAssignment(index, codeGenerator.performBINOP(index, one, ADD));

        codeGenerator.generateEnterBlock(loopBeginBlock);
        codeGenerator.setBuilderInsertionPoint(exitBlock);

        return newVector;
    } else if (tree->evaluatedType->vectorOrMatrixEnum == VECTOR) {
        auto mtree = std::dynamic_pointer_cast<MatrixTypeNode>(tree);

        mlir::Value row;
        mlir::Value column;

        // advanced ml algorithm for size inferencing
        if (mtree->sizeLeft) {
            row = std::any_cast<mlir::Value>(walk(mtree->sizeLeft));
        } else {
            std::vector<mlir::Value> arguments;
            arguments.push_back(*(this->inferenceContext.end() - 1));
            row = codeGenerator.generateCallNamed("rows", arguments);
        }

        if (mtree->sizeRight) {
            column = std::any_cast<mlir::Value>(walk(mtree->sizeRight));
        } else {
            std::vector<mlir::Value> arguments;
            arguments.push_back(*(this->inferenceContext.end() - 1));
            column = codeGenerator.generateCallNamed("columns", arguments);
        }

        // we do a little indexing
        auto rowIndex = codeGenerator.generateValue(0);

        auto one = codeGenerator.generateValue(1);

        auto matrix = codeGenerator.generateValue(row);

        rowIndex = codeGenerator.generateValue(0);

        mlir::Block* rowBeginBlock = codeGenerator.generateBlock();
        mlir::Block* rowTrueBlock = codeGenerator.generateBlock();
        mlir::Block* rowExitBlock = codeGenerator.generateBlock();

        codeGenerator.generateEnterBlock(rowBeginBlock);
        codeGenerator.setBuilderInsertionPoint(rowBeginBlock);

        auto inBoundsRow = codeGenerator.performBINOP(rowIndex, row, LTHAN);
        codeGenerator.generateCompAndJump(rowTrueBlock, rowExitBlock, codeGenerator.downcastToBool(inBoundsRow));
        codeGenerator.setBuilderInsertionPoint(rowTrueBlock);

        auto rowVec = codeGenerator.generateValue(column);

        auto colIndex = codeGenerator.generateValue(0);

        /* COL ========================= */
        mlir::Block* colBeginBlock = codeGenerator.generateBlock();
        mlir::Block* colTrueBlock = codeGenerator.generateBlock();
        mlir::Block* colExitBlock = codeGenerator.generateBlock();

        codeGenerator.generateEnterBlock(colBeginBlock);
        codeGenerator.setBuilderInsertionPoint(colBeginBlock);

        auto inBoundsCol = codeGenerator.performBINOP(colIndex, column, LTHAN);
        codeGenerator.generateCompAndJump(colTrueBlock, colExitBlock, codeGenerator.downcastToBool(inBoundsCol));

        codeGenerator.setBuilderInsertionPoint(colTrueBlock);

        auto result = codeGenerator.generateNullValue(mtree->evaluatedType);
        codeGenerator.appendCommon(rowVec, result);

        codeGenerator.generateAssignment(colIndex, codeGenerator.performBINOP(colIndex, one, ADD));
        codeGenerator.generateEnterBlock(colBeginBlock);
        codeGenerator.setBuilderInsertionPoint(colExitBlock);
        /* COL ========================= */

        codeGenerator.generateAssignment(rowIndex, codeGenerator.performBINOP(rowIndex, one, ADD));
        codeGenerator.appendCommon(matrix, rowVec);
        codeGenerator.generateEnterBlock(rowBeginBlock);
        codeGenerator.setBuilderInsertionPoint(rowExitBlock);

        return matrix;
    } else {

        switch (tree->evaluatedType->baseTypeEnum) {
        case TUPLE: {
            auto mtree = std::dynamic_pointer_cast<TupleTypeNode>(tree);
            int index = 1;
            std::vector<mlir::Value> children;
            for (auto child : mtree->getTypes()) {

                auto rhsItem = codeGenerator.indexCommonType(*(this->inferenceContext.end() - 1), codeGenerator.generateValue(index));

                this->inferenceContext.push_back(rhsItem);
                auto val = std::any_cast<mlir::Value>(walk(child));
                this->inferenceContext.pop_back();

                children.push_back(val);
                index++;
            }
            return codeGenerator.generateValue(children);
        }
        default:
            // base type, can resolve directly
            return codeGenerator.generateNullValue(tree->evaluatedType);
        }
    }
}
std::any BackendWalker::visitTypedef(std::shared_ptr<TypeDefNode> tree)
{
    return 0;
}

std::any BackendWalker::visitStreamOut(std::shared_ptr<StreamOut> tree)
{
    auto val = std::any_cast<mlir::Value>(walk(tree->getExpr()));

    this->codeGenerator.streamOut(val);

    return 0;
}

std::any BackendWalker::visitStreamIn(std::shared_ptr<StreamIn> tree)
{
    auto val = std::any_cast<mlir::Value>(walk(tree->getExpr()));

    this->codeGenerator.streamIn(val);

    return 0;
}

// === EXPRESSION AST NODES ===
std::any BackendWalker::visitID(std::shared_ptr<IDNode> tree)
{
    if (tree->sym->scope->getScopeName() == "Global") {
        return codeGenerator.generateLoadIdentifier(tree->sym->mlirName);
    }

    // might be arg
    if (tree->sym->index >= 0) {
        auto val = codeGenerator.generateLoadArgument(tree->sym->index);

        this->inferenceContext.push_back(val);
        auto toType = std::any_cast<mlir::Value>(walk(tree->sym->typeNode));
        this->inferenceContext.pop_back();

        if (this->fetchRaw) {
            return val;
        } else {
            return codeGenerator.promotion(val, toType);
        }
    } else if (tree->sym->declarationIndex >= 0) {
        return codeGenerator.indexCommonType(*(this->variableStack.end() - 1 - tree->numStackBehind), codeGenerator.generateValue(tree->sym->declarationIndex));
    }
}

std::any BackendWalker::visitIdentity(std::shared_ptr<IdentityNode> tree)
{
    return codeGenerator.generateIdentityValue(tree->evaluatedType);
}

std::any BackendWalker::visitNull(std::shared_ptr<NullNode> tree)
{
    return codeGenerator.generateNullValue(tree->evaluatedType);
}

std::any BackendWalker::visitInt(std::shared_ptr<IntNode> tree)
{
    return codeGenerator.generateValue(tree->getVal());
}

std::any BackendWalker::visitReal(std::shared_ptr<RealNode> tree)
{
    return codeGenerator.generateValue(tree->getVal());
}

std::any BackendWalker::visitChar(std::shared_ptr<CharNode> tree)
{
    return codeGenerator.generateValue(tree->getVal());
}

std::any BackendWalker::visitBool(std::shared_ptr<BoolNode> tree)
{
    return codeGenerator.generateValue(tree->getVal());
}

std::any BackendWalker::visitVector(std::shared_ptr<VectorNode> tree)
{
    std::vector<mlir::Value> values;

    for (auto node : tree->getElements()) {
        values.push_back(std::any_cast<mlir::Value>(walk(node)));
    }

    auto result = codeGenerator.generateValue(values);
    codeGenerator.normalize(result);
    return result;
}

std::any BackendWalker::visitString(std::shared_ptr<VectorNode> tree)
{
    return codeGenerator.generateValue(tree->getVal());
}

std::any BackendWalker::visitTuple(std::shared_ptr<TupleNode> tree)
{
    std::vector<mlir::Value> values;

    for (auto node : tree->getVal()) {
        values.push_back(std::any_cast<mlir::Value>(walk(node)));
    }

    return codeGenerator.generateValue(values);
}

std::any BackendWalker::visitTupleIndex(std::shared_ptr<TupleIndexNode> tree)
{
    mlir::Value indexee = std::any_cast<mlir::Value>(walk(tree->getIDNode()));

    return codeGenerator.indexCommonType(indexee, codeGenerator.generateValue(tree->index + 1));
}

std::any BackendWalker::visitStdInputNode(std::shared_ptr<StdInputNode> tree)
{
    return codeGenerator.getStreamStateVar();
}

// Expr/Binary
std::any BackendWalker::visitCast(std::shared_ptr<CastNode> tree)
{
    auto val = std::any_cast<mlir::Value>(walk(tree->getExpr()));

    this->inferenceContext.push_back(val);
    auto type = std::any_cast<mlir::Value>(walk(tree->getType()));
    this->inferenceContext.pop_back();

    return codeGenerator.cast(val, type);
}

std::any BackendWalker::visitArith(std::shared_ptr<BinaryArithNode> tree)
{
    auto lhs = std::any_cast<mlir::Value>(walk(tree->getLHS()));
    auto rhs = std::any_cast<mlir::Value>(walk(tree->getRHS()));

    return codeGenerator.performBINOP(lhs, rhs, tree->op);
}

std::any BackendWalker::visitUnaryArith(std::shared_ptr<UnaryArithNode> tree)
{
    auto expr = std::any_cast<mlir::Value>(walk(tree->getExpr()));
    return codeGenerator.performUNARYOP(expr, tree->op);
}

std::any BackendWalker::visitCmp(std::shared_ptr<BinaryCmpNode> tree)
{
    auto lhs = std::any_cast<mlir::Value>(walk(tree->getLHS()));
    auto rhs = std::any_cast<mlir::Value>(walk(tree->getRHS()));

    return codeGenerator.performBINOP(lhs, rhs, tree->op);
}

std::any BackendWalker::visitIndex(std::shared_ptr<IndexNode> tree)
{
    auto indexee = std::any_cast<mlir::Value>(walk(tree->getIndexee()));

    // im sad we don't use recursion more
    for (int i = 1; i < tree->children.size(); i++) {
        auto indexor = std::any_cast<mlir::Value>(walk(tree->children[i]));
        indexee = codeGenerator.indexCommonType(indexee, indexor);
    }

    return indexee;
}

// Expr/Vector
std::any BackendWalker::visitFilter(std::shared_ptr<FilterNode> tree)
{
    // what we are filtering from
    auto filteree = std::any_cast<mlir::Value>(walk(tree->getDomain()));
    auto one = codeGenerator.generateValue(1);

    auto stack = codeGenerator.initializeStack(1);
    this->variableStack.push_back(stack);

    // max amount of filters
    auto maxFiltered = codeGenerator.generateValue((int)tree->getExprList().size() + 1);

    // empty filter we are appending to
    auto filter = codeGenerator.generateValue(maxFiltered);

    std::vector<mlir::Value> argument;
    argument.push_back(filteree);

    auto maxVectorSize = codeGenerator.generateCallNamed("length", argument);

    // +1 for left over
    for (int i = 0; i < tree->getExprList().size() + 1; i++) {
        auto vector = codeGenerator.generateValue(maxVectorSize);
        codeGenerator.appendCommon(filter, vector);
    }

    mlir::Block* loopBeginBlock = codeGenerator.generateBlock();
    mlir::Block* trueBlock = codeGenerator.generateBlock();
    mlir::Block* exitBlock = codeGenerator.generateBlock();
    auto index = codeGenerator.generateValue(1);

    codeGenerator.generateEnterBlock(loopBeginBlock);
    codeGenerator.setBuilderInsertionPoint(loopBeginBlock);

    auto inBounds = codeGenerator.performBINOP(index, maxVectorSize, LEQ);

    codeGenerator.generateCompAndJump(trueBlock, exitBlock, codeGenerator.downcastToBool(inBounds));

    codeGenerator.setBuilderInsertionPoint(trueBlock);

    auto indexedVal = codeGenerator.indexCommonType(filteree, index);
    codeGenerator.generateAssignment(codeGenerator.indexCommonType(stack, one), indexedVal);

    auto appended = codeGenerator.generateValue(false);
    for (int i = 0; i < tree->getExprList().size(); i++) {

        auto result = std::any_cast<mlir::Value>(walk(tree->getExprList()[i]));

        mlir::Block* trueResult = codeGenerator.generateBlock();
        mlir::Block* falseResult = codeGenerator.generateBlock();
        codeGenerator.generateCompAndJump(trueResult, falseResult, codeGenerator.downcastToBool(result));

        codeGenerator.setBuilderInsertionPoint(trueResult);

        codeGenerator.appendCommon(codeGenerator.indexCommonType(filter, codeGenerator.generateValue(i + 1)), indexedVal);
        codeGenerator.generateAssignment(appended, codeGenerator.performBINOP(appended, codeGenerator.generateValue(true), OR));

        codeGenerator.generateEnterBlock(falseResult);
        codeGenerator.setBuilderInsertionPoint(falseResult);
    }

    mlir::Block* satisfied = codeGenerator.generateBlock();
    mlir::Block* unsatisfied = codeGenerator.generateBlock();
    codeGenerator.generateCompAndJump(satisfied, unsatisfied, codeGenerator.downcastToBool(appended));

    codeGenerator.setBuilderInsertionPoint(unsatisfied);
    codeGenerator.appendCommon(codeGenerator.indexCommonType(filter, maxFiltered), codeGenerator.indexCommonType(filteree, index));

    codeGenerator.generateEnterBlock(satisfied);
    codeGenerator.setBuilderInsertionPoint(satisfied);

    codeGenerator.generateAssignment(index, codeGenerator.performBINOP(index, one, ADD));

    codeGenerator.generateEnterBlock(loopBeginBlock);
    codeGenerator.setBuilderInsertionPoint(exitBlock);

    this->variableStack.pop_back();
    return filter;
}

std::any BackendWalker::visitGenerator(std::shared_ptr<GeneratorNode> tree)
{

    if (tree->getVectDomain()) {
        auto stack = codeGenerator.initializeStack(1);
        auto baseVec = std::any_cast<mlir::Value>(walk(tree->getVectDomain()));

        // we do a little indexing
        auto index = codeGenerator.generateValue(1);

        auto one = codeGenerator.generateValue(1);

        this->variableStack.push_back(stack);

        // build arg list
        std::vector<mlir::Value> argument;
        argument.push_back(baseVec);

        auto length = codeGenerator.generateCallNamed("length", argument);
        auto generatorVector = codeGenerator.generateValue(length);

        mlir::Block* loopBeginBlock = codeGenerator.generateBlock();
        mlir::Block* trueBlock = codeGenerator.generateBlock();
        mlir::Block* exitBlock = codeGenerator.generateBlock();

        codeGenerator.generateEnterBlock(loopBeginBlock);
        codeGenerator.setBuilderInsertionPoint(loopBeginBlock);

        auto inBounds = codeGenerator.performBINOP(index, length, LEQ);

        codeGenerator.generateCompAndJump(trueBlock, exitBlock, codeGenerator.downcastToBool(inBounds));

        codeGenerator.setBuilderInsertionPoint(trueBlock);
        auto indexedVal = codeGenerator.indexCommonType(baseVec, index);
        codeGenerator.generateAssignment(codeGenerator.indexCommonType(stack, one), indexedVal);

        auto result = std::any_cast<mlir::Value>(walk(tree->getExpr()));

        codeGenerator.appendCommon(generatorVector, result);
        codeGenerator.generateAssignment(index, codeGenerator.performBINOP(index, one, ADD));

        codeGenerator.generateEnterBlock(loopBeginBlock);
        codeGenerator.setBuilderInsertionPoint(exitBlock);

        this->variableStack.pop_back();
        return generatorVector;
    } else {
        auto row = std::any_cast<mlir::Value>(walk(tree->getMatrixDomain().first));
        auto column = std::any_cast<mlir::Value>(walk(tree->getMatrixDomain().second));

        // we do a little indexing
        auto stack = codeGenerator.initializeStack(2);

        auto rowIndex = codeGenerator.generateValue(1);
        auto rowDomain = codeGenerator.generateValue(1);
        auto colDomain = codeGenerator.generateValue(1);

        auto one = codeGenerator.generateValue(1);
        auto two = codeGenerator.generateValue(2);

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

        mlir::Block* matrixBeginBlock = codeGenerator.generateBlock();
        mlir::Block* matrixTrueBlock = codeGenerator.generateBlock();
        mlir::Block* matrixExitBlock = codeGenerator.generateBlock();

        codeGenerator.generateEnterBlock(matrixBeginBlock);
        codeGenerator.setBuilderInsertionPoint(matrixBeginBlock);
        auto inBoundsRow = codeGenerator.performBINOP(rowIndex, rowLength, LEQ);
        codeGenerator.generateCompAndJump(matrixTrueBlock, matrixExitBlock, codeGenerator.downcastToBool(inBoundsRow));
        codeGenerator.setBuilderInsertionPoint(matrixTrueBlock);

        codeGenerator.appendCommon(generatorVector, codeGenerator.generateValue(colLength));
        codeGenerator.generateAssignment(rowIndex, codeGenerator.performBINOP(rowIndex, one, ADD));

        rowIndex = codeGenerator.generateValue(1);

        mlir::Block* rowBeginBlock = codeGenerator.generateBlock();
        mlir::Block* rowTrueBlock = codeGenerator.generateBlock();
        mlir::Block* rowExitBlock = codeGenerator.generateBlock();

        codeGenerator.generateEnterBlock(rowBeginBlock);
        codeGenerator.setBuilderInsertionPoint(rowBeginBlock);

        inBoundsRow = codeGenerator.performBINOP(rowIndex, rowLength, LEQ);
        codeGenerator.generateCompAndJump(rowTrueBlock, rowExitBlock, codeGenerator.downcastToBool(inBoundsRow));
        codeGenerator.setBuilderInsertionPoint(rowTrueBlock);
        auto colIndex = codeGenerator.generateValue(1);

        /* COL ========================= */
        mlir::Block* colBeginBlock = codeGenerator.generateBlock();
        mlir::Block* colTrueBlock = codeGenerator.generateBlock();
        mlir::Block* colExitBlock = codeGenerator.generateBlock();

        codeGenerator.generateEnterBlock(colBeginBlock);
        codeGenerator.setBuilderInsertionPoint(colBeginBlock);

        auto inBoundsCol = codeGenerator.performBINOP(colIndex, colLength, LEQ);
        codeGenerator.generateCompAndJump(colTrueBlock, colExitBlock, codeGenerator.downcastToBool(inBoundsCol));

        codeGenerator.setBuilderInsertionPoint(colTrueBlock);

        auto indexedRow = codeGenerator.indexCommonType(row, rowIndex);
        auto indexedCol = codeGenerator.indexCommonType(column, colIndex);

        codeGenerator.generateAssignment(codeGenerator.indexCommonType(stack, one), indexedRow);
        codeGenerator.generateAssignment(codeGenerator.indexCommonType(stack, two), indexedCol);

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

std::any BackendWalker::visitRangeVec(std::shared_ptr<RangeVecNode> tree)
{
    auto lower = std::any_cast<mlir::Value>(walk(tree->getStart()));
    auto upper = std::any_cast<mlir::Value>(walk(tree->getEnd()));

    return codeGenerator.generateValue(lower, upper);
}

// === BLOCK AST NODES ===

std::any BackendWalker::visitConditional(std::shared_ptr<ConditionalNode> tree)
{
    // create all our block ahead of time
    // for each condition, we need a true and a false mlir::Block.
    // these go at the beginning and end of each if statement's body, respectively
    // finally, we need an end mlir::Block placed at the end of all conditions
    std::vector<mlir::Block*> trueBlocks;
    std::vector<mlir::Block*> falseBlocks;
    for (int i = 0; i < tree->conditions.size(); i++) {
        trueBlocks.push_back(codeGenerator.generateBlock());
        falseBlocks.push_back(codeGenerator.generateBlock());
    }
    mlir::Block* endBlock = codeGenerator.generateBlock();

    bool allReturn = true;

    // now, go through the conditions and bodies and generate the code
    // the number of bodies is never less than the number of conditions
    for (int i = 0; i < tree->conditions.size(); i++) {
        auto condResultVal = std::any_cast<mlir::Value>(walk(tree->conditions[i]));
        auto condResultBool = codeGenerator.downcastToBool(condResultVal);

        codeGenerator.generateCompAndJump(trueBlocks[i], falseBlocks[i], condResultBool);

        codeGenerator.setBuilderInsertionPoint(trueBlocks[i]);
        walk(tree->bodies[i]);

        // return was dropped during walk, don't need to bound back
        if (!this->returnDropped)
            codeGenerator.conditionalJumpToBlock(endBlock, !earlyReturn);
        allReturn = allReturn & returnDropped;

        this->returnDropped = false;
        this->earlyReturn = false;

        codeGenerator.setBuilderInsertionPoint(falseBlocks[i]);
    }

    // if there is an "else" clause, we will have one more "body" node
    if (tree->bodies.size() > tree->conditions.size()) {
        this->returnDropped = false;
        walk(tree->bodies[tree->bodies.size() - 1]);

        allReturn = allReturn & returnDropped;
    } else {
        allReturn = false;
    }

    if (!this->returnDropped) {
        codeGenerator.conditionalJumpToBlock(endBlock, !earlyReturn);
    }

    if (allReturn) {
        endBlock->erase();
    } else {
        codeGenerator.setBuilderInsertionPoint(endBlock);
    }
    this->returnDropped = allReturn & returnDropped;

    // note returnDropped isn't turned off. dropping a return in an else
    // guarantees early return. stop generating code, it will be unreachable
    this->earlyReturn = false;

    return 0;
}

std::any BackendWalker::visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree)
{
    auto loopBody = codeGenerator.generateBlock(); // start of loop
    auto loopExit = codeGenerator.generateBlock(); // the rest of the program

    this->loopBlocks.push_back(std::make_pair(loopBody, loopExit));

    // body of loop
    codeGenerator.generateEnterBlock(loopBody);
    codeGenerator.setBuilderInsertionPoint(loopBody);
    walk(tree->getBody());

    // loop exit
    if (!this->returnDropped) {
        codeGenerator.generateEnterBlock(loopBody);
        codeGenerator.setBuilderInsertionPoint(loopExit);
    } else {
        loopExit->erase();
    }
    this->loopBlocks.pop_back();

    this->earlyReturn = false;

    return 0;
}

std::any BackendWalker::visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree)
{
    auto loopCheck = codeGenerator.generateBlock(); // check
    auto loopBody = codeGenerator.generateBlock(); // body
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
    if (!this->returnDropped)
        codeGenerator.generateEnterBlock(loopCheck);
    this->returnDropped = false;

    // loop exit
    codeGenerator.setBuilderInsertionPoint(loopExit);
    this->loopBlocks.pop_back();

    return 0;
}

std::any BackendWalker::visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree)
{
    // check goes after the body
    auto loopBody = codeGenerator.generateBlock();
    auto loopCheck = codeGenerator.generateBlock();
    auto loopExit = codeGenerator.generateBlock();

    this->loopBlocks.push_back(std::make_pair(loopBody, loopExit));

    // body of loop
    codeGenerator.generateEnterBlock(loopBody);
    codeGenerator.setBuilderInsertionPoint(loopBody);
    walk(tree->getBody());
    if (!this->returnDropped)
        codeGenerator.generateEnterBlock(loopCheck);
    this->returnDropped = false;

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

std::any BackendWalker::visitIteratorLoop(std::shared_ptr<IteratorLoopNode> tree)
{
    // important loop info we need to have
    // tuple of <loopStart, loopExit>
    std::vector<std::pair<mlir::Block*, mlir::Block*>> blocks;
    auto stack = codeGenerator.initializeStack(tree->getDomainExprs().size());
    // create new nested loop for each domainExpr
    int i = 1;
    for (auto& domainExpr : tree->getDomainExprs()) {
        auto domainNode = domainExpr.second;
        auto domainSym = domainExpr.first;

        // create/load domain vector
        auto domain = std::any_cast<mlir::Value>(walk(domainNode));

        // var to index the domain
        auto domainIdx = codeGenerator.generateValue(1);
        auto domainIdxVal = codeGenerator.generateNullValue(domainSym->typeSym);
        codeGenerator.generateAssignment(codeGenerator.indexCommonType(stack, codeGenerator.generateValue(i)), domainIdxVal);

        // get length of domainVec
        auto domainLength = codeGenerator.generateCallNamed("length", { domain });

        // START THE LOOP
        mlir::Block* loopBeginBlock = codeGenerator.generateBlock();
        mlir::Block* trueBlock = codeGenerator.generateBlock();
        mlir::Block* exitBlock = codeGenerator.generateBlock();

        blocks.push_back(std::make_pair(loopBeginBlock, exitBlock));

        // PREDICATE (domainIdx < length)
        codeGenerator.generateEnterBlock(loopBeginBlock);
        codeGenerator.setBuilderInsertionPoint(loopBeginBlock);
        auto inBounds = codeGenerator.performBINOP(domainIdx, domainLength, LEQ);
        codeGenerator.generateCompAndJump(trueBlock, exitBlock, codeGenerator.downcastToBool(inBounds));

        // BODY (true block)
        codeGenerator.setBuilderInsertionPoint(trueBlock);
        // set domainIdxVal to domain[domainIdx]
        auto indexedVal = codeGenerator.indexCommonType(domain, domainIdx);
        codeGenerator.generateAssignment(codeGenerator.indexCommonType(stack, codeGenerator.generateValue(i)), indexedVal);

        // increment domainIdx
        // doing this here so if there is a `break` of `continue` stmt in the body, we won't be stuck in an infinite loop
        auto one = codeGenerator.generateValue(1);
        auto newDomainIdx = codeGenerator.performBINOP(domainIdx, one, ADD);
        codeGenerator.generateAssignment(domainIdx, newDomainIdx);
        i += 1;
    }

    // although the iterator loop can be split into multiple loop, it is in essence only one singular loop
    // if there is a break/continue statement, we need to jump to the correct exit block
    this->loopBlocks.push_back(std::make_pair(blocks[blocks.size() - 1].first, blocks[blocks.size() - 1].second));

    this->variableStack.push_back(stack);

    // walk the body
    walk(tree->getBody());

    this->variableStack.pop_back();

    // add all exitBlocks, increment domainIdx
    // reverse it first; we need to traverse in reverse order
    std::reverse(blocks.begin(), blocks.end());
    for (auto& blockInfo : blocks) {
        auto enter = blockInfo.first;
        auto exit = blockInfo.second;

        // if there is a return, we only skip the first layer of block jumps back (it seems to work?)
        if (!this->returnDropped)
            codeGenerator.generateEnterBlock(enter);
        this->returnDropped = false;

        codeGenerator.setBuilderInsertionPoint(exit);
    }
    this->loopBlocks.pop_back();

    return 0;
}

std::any BackendWalker::visitBreak(std::shared_ptr<BreakNode> tree)
{
    if (this->loopBlocks.empty()) {
        throw StatementError(tree->loc(), "Break statement outside of loop");
    }

    auto loopExit = this->loopBlocks.back().second;
    codeGenerator.generateEnterBlock(loopExit);
    this->earlyReturn = true;

    return 0;
}

std::any BackendWalker::visitContinue(std::shared_ptr<ContinueNode> tree)
{
    if (this->loopBlocks.empty()) {
        throw StatementError(tree->loc(), "Continue statement outside of loop");
    }

    auto loopBody = this->loopBlocks.back().first;
    codeGenerator.generateEnterBlock(loopBody);
    this->earlyReturn = true;

    return 0;
}

std::any BackendWalker::visitProcedure(std::shared_ptr<ProcedureNode> tree)
{
    if (tree->body) {
        // for now we don't proper return values, assume everything void
        auto block = codeGenerator.generateFunctionDefinition(tree->nameSym->name,
            tree->orderedArgs.size(),
            false);

        this->returnType = tree->getRetTypeNode();

        walk(tree->body);

        codeGenerator.generateEndFunctionDefinition(block, tree->loc());
        codeGenerator.verifyFunction(tree->loc(), "Procedure " + tree->nameSym->name);
        this->returnDropped = false;
    }

    return 0;
}

std::any BackendWalker::visitFunction(std::shared_ptr<FunctionNode> tree)
{
    if (tree->body) {

        auto block = codeGenerator.generateFunctionDefinition(tree->funcNameSym->name,
            tree->orderedArgs.size(),
            false);
        // stack
        this->returnType = tree->getRetTypeNode();

        walk(tree->body);

        codeGenerator.generateEndFunctionDefinition(block, tree->loc());
        codeGenerator.verifyFunction(tree->loc(), "Function " + tree->funcNameSym->name);
        this->returnDropped = false;
    }
    return 0;
}

std::any BackendWalker::visitCall(std::shared_ptr<CallNode> tree)
{
    std::vector<mlir::Value> arguments;

    for (auto argument : tree->children) {
        arguments.push_back(std::any_cast<mlir::Value>(walk(argument)));
    }

    std::string funcSymbol = (tree->MethodRef->isBuiltIn()) ? tree->MethodRef->name : tree->CallName->name;
    return codeGenerator.generateCallNamed(funcSymbol, arguments);
}

std::any BackendWalker::visitReturn(std::shared_ptr<ReturnNode> tree)
{
    auto returnValue = tree->returnExpr ? std::any_cast<mlir::Value>(walk(tree->returnExpr)) : codeGenerator.generateValue(0);

    if (tree->returnExpr) {
        this->inferenceContext.push_back(returnValue);
        auto toType = std::any_cast<mlir::Value>(walk(this->returnType));
        returnValue = codeGenerator.promotion(returnValue, toType);
        this->inferenceContext.pop_back();
    } else {
        returnValue = codeGenerator.generateValue(0);
    }

    codeGenerator.generateReturn(returnValue, this->variableStack.back());

    this->returnDropped = true;
    return 0;
}

std::any BackendWalker::visitBlock(std::shared_ptr<BlockNode> tree)
{
    codeGenerator.pushScope();

    this->variableStack.push_back(codeGenerator.initializeStack(tree->scope->numVarDeclared));
    auto returnVal = walkChildren(tree);

    // cheeky return. catches void functions + generalizes if/else returns
    if (!returnDropped && !this->returnType) {
        codeGenerator.generateReturn(codeGenerator.generateValue(0), this->variableStack.back());
    }
    this->variableStack.pop_back();

    codeGenerator.popScope();
    return returnVal;
}

std::any BackendWalker::visitConcat(std::shared_ptr<ConcatNode> tree)
{
    auto lhs = std::any_cast<mlir::Value>(walk(tree->getLHS()));
    auto rhs = std::any_cast<mlir::Value>(walk(tree->getRHS()));

    return codeGenerator.performBINOP(lhs, rhs, CONCAT);
}

std::any BackendWalker::visitStride(std::shared_ptr<StrideNode> tree)
{
    auto lhs = std::any_cast<mlir::Value>(walk(tree->getLHS()));
    auto rhs = std::any_cast<mlir::Value>(walk(tree->getRHS()));

    return codeGenerator.performBINOP(lhs, rhs, STRIDE);
}
