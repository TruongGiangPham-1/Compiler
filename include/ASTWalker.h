#pragma once
#include "ASTNode/ASTNode.h"
#include "ASTNode/ArgNode.h"
#include "ASTNode/AssignNode.h"
#include "ASTNode/Block/ConditionalNode.h"
#include "ASTNode/BreakNode.h"
#include "ASTNode/CallNode.h"
#include "ASTNode/ContinueNode.h"
#include "ASTNode/DeclNode.h"
#include "ASTNode/Expr/Binary/BinaryExpr.h"
#include "ASTNode/Expr/CastNode.h"
#include "ASTNode/Expr/ExprListNode.h"
#include "ASTNode/Expr/IdentityNode.h"
#include "ASTNode/Expr/Literal/BoolNode.h"
#include "ASTNode/Expr/Literal/CharNode.h"
#include "ASTNode/Expr/Literal/IDNode.h"
#include "ASTNode/Expr/Literal/IntNode.h"
#include "ASTNode/Expr/Literal/MatrixNode.h"
#include "ASTNode/Expr/Literal/RealNode.h"
#include "ASTNode/Expr/Literal/StringNode.h"
#include "ASTNode/Expr/Literal/TupleNode.h"
#include "ASTNode/Expr/Literal/VectorNode.h"
#include "ASTNode/Expr/NullNode.h"
#include "ASTNode/Expr/StdInputNode.h"
#include "ASTNode/Expr/TupleIndexNode.h"
#include "ASTNode/Expr/Unary/UnaryExpr.h"
#include "ASTNode/Expr/Vector/FilterNode.h"
#include "ASTNode/Expr/Vector/GeneratorNode.h"
#include "ASTNode/Expr/Vector/RangeVecNode.h"
#include "ASTNode/Loop/InfiniteLoopNode.h"
#include "ASTNode/Loop/IteratorLoopNode.h"
#include "ASTNode/Loop/PostPredicatedLoopNode.h"
#include "ASTNode/Loop/PredicatedLoopNode.h"
#include "ASTNode/Method/FunctionNode.h"
#include "ASTNode/Method/ProcedureNode.h"
#include "ASTNode/Method/ReturnNode.h"
#include "ASTNode/Stream/StreamIn.h"
#include "ASTNode/Stream/StreamOut.h"
#include "ASTNode/Type/MatrixTypeNode.h"
#include "ASTNode/Type/TypeNode.h"
#include "ASTNode/Type/VectorTypeNode.h"
#include "ASTNode/TypeDefNode.h"

namespace gazprea {
class ASTWalker {
protected:
    virtual std::any walkChildren(std::shared_ptr<ASTNode> tree);

public:
    ASTWalker() {};

    virtual std::any walk(std::shared_ptr<ASTNode> tree);

    // === TOP LEVEL AST NODES ===
    virtual std::any visitAssign(std::shared_ptr<AssignNode> tree);
    virtual std::any visitDecl(std::shared_ptr<DeclNode> tree);
    virtual std::any visitStreamOut(std::shared_ptr<StreamOut> tree);
    virtual std::any visitStreamIn(std::shared_ptr<StreamIn> tree);

    // resolve these
    virtual std::any visitType(std::shared_ptr<TypeNode> tree);
    virtual std::any visitTypedef(std::shared_ptr<TypeDefNode> tree);

    // === EXPRESSION AST NODES ===
    virtual std::any visitExpressionList(std::shared_ptr<ExprListNode> tree);
    virtual std::any visitID(std::shared_ptr<IDNode> tree);
    virtual std::any visitInt(std::shared_ptr<IntNode> tree);
    virtual std::any visitReal(std::shared_ptr<RealNode> tree);
    virtual std::any visitTuple(std::shared_ptr<TupleNode> tree);
    virtual std::any visitChar(std::shared_ptr<CharNode> tree);
    virtual std::any visitString(std::shared_ptr<VectorNode> tree);
    virtual std::any visitBool(std::shared_ptr<BoolNode> tree);
    virtual std::any visitTupleIndex(std::shared_ptr<TupleIndexNode> tree);
    virtual std::any visitVector(std::shared_ptr<VectorNode> tree);
    // virtual std::any visitMatrix(std::shared_ptr<MatrixNode> tree);
    virtual std::any visitStdInputNode(std::shared_ptr<StdInputNode> tree);

    // Expr/Binary

    virtual std::any visitArith(std::shared_ptr<BinaryArithNode> tree);
    virtual std::any visitCmp(std::shared_ptr<BinaryCmpNode> tree);
    virtual std::any visitIndex(std::shared_ptr<IndexNode> tree);
    virtual std::any visitCast(std::shared_ptr<CastNode> tree);
    virtual std::any visitUnaryArith(std::shared_ptr<UnaryArithNode> tree);
    virtual std::any visitConcat(std::shared_ptr<ConcatNode> tree);
    virtual std::any visitStride(std::shared_ptr<StrideNode> tree);
    // Expr/Vector
    virtual std::any visitFilter(std::shared_ptr<FilterNode> tree);
    virtual std::any visitGenerator(std::shared_ptr<GeneratorNode> tree);
    virtual std::any visitRangeVec(std::shared_ptr<RangeVecNode> tree);

    // == BLock
    virtual std::any visitBlock(std::shared_ptr<BlockNode> tree);
    // === BLOCK AST NODES ===
    virtual std::any visitConditional(std::shared_ptr<ConditionalNode> tree);
    virtual std::any visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree);
    virtual std::any visitIteratorLoop(std::shared_ptr<IteratorLoopNode> tree);
    virtual std::any visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree);
    virtual std::any visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree);
    virtual std::any visitBreak(std::shared_ptr<BreakNode> tree);
    virtual std::any visitContinue(std::shared_ptr<ContinueNode> tree);

    // === BLOCK FUNCTION NODES ===
    virtual std::any visitFunction(std::shared_ptr<FunctionNode> tree);
    virtual std::any visitCall(std::shared_ptr<CallNode> tree);
    // === BlOCK PROCEDURE NODES ===
    virtual std::any visitProcedure(std::shared_ptr<ProcedureNode> tree);
    virtual std::any visitReturn(std::shared_ptr<ReturnNode> tree);
    virtual std::any visitParameter(std::shared_ptr<ArgNode> tree);
    // === FUNCTION CALL NODE ===

    // === Null and identity
    virtual std::any visitNull(std::shared_ptr<NullNode> tree);
    virtual std::any visitIdentity(std::shared_ptr<IdentityNode> tree);
};
}
