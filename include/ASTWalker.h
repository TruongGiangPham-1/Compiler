#pragma once
#include "ASTNode/ASTNode.h"
#include "ASTNode/ArgNode.h"
#include "ASTNode/AssignNode.h"
#include "ASTNode/Expr/Literal/BoolNode.h"
#include "ASTNode/Type/TypeNode.h"
#include "ASTNode/DeclNode.h"
#include "ASTNode/Stream/StreamOut.h"
#include "ASTNode/Expr/Literal/IDNode.h"
#include "ASTNode/Expr/Literal/IntNode.h"
#include "ASTNode/Expr/Literal/RealNode.h"
#include "ASTNode/Expr/Literal/CharNode.h"
#include "ASTNode/Expr/Literal/TupleNode.h"
#include "ASTNode/Method/ProcedureNode.h"
#include "ASTNode/Expr/Binary/BinaryExpr.h"
#include "ASTNode/Expr/Vector/RangeVecNode.h"
#include "ASTNode/Expr/Vector/GeneratorNode.h"
#include "ASTNode/Expr/Vector/FilterNode.h"
#include "ASTNode/Expr/Unary/UnaryExpr.h"
#include "ASTNode/Loop/PredicatedLoopNode.h"
#include "ASTNode/Loop/PostPredicatedLoopNode.h"
#include "ASTNode/Loop/InfiniteLoopNode.h"
#include "ASTNode/BreakNode.h"
#include "ASTNode/ContinueNode.h"
#include "ASTNode/Block/ConditionalNode.h"
#include "ASTNode/Method/FunctionNode.h"
#include "ASTNode/FunctionCallNode.h"

namespace gazprea {
    class ASTWalker {
    protected:
        std::any walkChildren(std::shared_ptr<ASTNode> tree);

    public:
        ASTWalker() {};

        std::any walk(std::shared_ptr<ASTNode> tree);

        // === TOP LEVEL AST NODES ===
        virtual std::any visitAssign(std::shared_ptr<AssignNode> tree);
        virtual std::any visitDecl(std::shared_ptr<DeclNode> tree);
        virtual std::any visitPrint(std::shared_ptr<StreamOut> tree);

        // resolve these
        virtual std::any visitType(std::shared_ptr<TypeNode> tree);

        // === EXPRESSION AST NODES ===
        virtual std::any visitID(std::shared_ptr<IDNode> tree);
        virtual std::any visitInt(std::shared_ptr<IntNode> tree);
        virtual std::any visitReal(std::shared_ptr<RealNode> tree);
        virtual std::any visitTuple(std::shared_ptr<TupleNode> tree);
        virtual std::any visitChar(std::shared_ptr<CharNode> tree);
        virtual std::any visitBool(std::shared_ptr<BoolNode> tree);

        // Expr/Binary

        virtual std::any visitArith(std::shared_ptr<BinaryArithNode> tree);
        virtual std::any visitCmp(std::shared_ptr<BinaryCmpNode> tree);
        virtual std::any visitIndex(std::shared_ptr<IndexNode> tree);
        virtual std::any visitUnaryArith(std::shared_ptr<UnaryArithNode>tree);
        // Expr/Vector
        virtual std::any visitFilter(std::shared_ptr<FilterNode> tree);
        virtual std::any visitGenerator(std::shared_ptr<GeneratorNode> tree);
        virtual std::any visitRangeVec(std::shared_ptr<RangeVecNode> tree);


        // == BLock
        virtual std::any visitBlock(std::shared_ptr<BlockNode>tree);
        // === BLOCK AST NODES ===
        virtual std::any visitConditional(std::shared_ptr<ConditionalNode> tree);
        virtual std::any visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree);
        virtual std::any visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree);
        virtual std::any visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree);
        virtual std::any visitBreak(std::shared_ptr<BreakNode> tree);
        virtual std::any visitContinue(std::shared_ptr<ContinueNode> tree);

        // === BLOCK FUNCTION NODES ===
        virtual std::any visitFunction(std::shared_ptr<FunctionNode> tree);
        virtual std::any visitFunctionCall(std::shared_ptr<FunctionCallNode> tree);
        // === BlOCK PROCEDURE NODES ===
        virtual std::any visitProcedure(std::shared_ptr<ProcedureNode> tree);
        virtual std::any visitParameter(std::shared_ptr<ArgNode> tree);


        // === FUNCTION CALL NODE ===

    };
}
