#pragma once
#include "ASTNode/ASTNode.h"
#include "ASTNode/AssignNode.h"
#include "ASTNode/TypeNode.h"
#include "ASTNode/DeclNode.h"
#include "ASTNode/PrintNode.h"
#include "ASTNode/Expr/IDNode.h"
#include "ASTNode/Expr/IntNode.h"
#include "ASTNode/Expr/Binary/BinaryExpr.h"
#include "ASTNode/Expr/Vector/RangeVecNode.h"
#include "ASTNode/Expr/Vector/GeneratorNode.h"
#include "ASTNode/Expr/Vector/FilterNode.h"
#include "ASTNode/Expr/Unary/UnaryExpr.h"
#include "ASTNode/Block/LoopNode.h"
#include "ASTNode/Block/ConditionalNode.h"
#include "ASTNode/Block/FunctionNode.h"
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
        virtual std::any visitPrint(std::shared_ptr<PrintNode> tree);
        virtual std::any visitType(std::shared_ptr<TypeNode> tree);

        // === EXPRESSION AST NODES ===
        virtual std::any visitID(std::shared_ptr<IDNode> tree);
        virtual std::any visitInt(std::shared_ptr<IntNode> tree);
        // Expr/Binary
        virtual std::any visitArith(std::shared_ptr<ArithNode> tree);
        virtual std::any visitCmp(std::shared_ptr<CmpNode> tree);
        virtual std::any visitIndex(std::shared_ptr<IndexNode> tree);
        // Expr/Vector
        virtual std::any visitFilter(std::shared_ptr<FilterNode> tree);
        virtual std::any visitGenerator(std::shared_ptr<GeneratorNode> tree);
        virtual std::any visitRangeVec(std::shared_ptr<RangeVecNode> tree);

        // === BLOCK AST NODES ===
        virtual std::any visitConditional(std::shared_ptr<ConditionalNode> tree);
        virtual std::any visitLoop(std::shared_ptr<LoopNode> tree);
        // === BLOCK FUNCTION NODES ===
        virtual std::any visitFunctionForward(std::shared_ptr<FunctionForwardNode> tree);
        virtual std::any visitFunctionSingle(std::shared_ptr<FunctionSingleNode> tree);
        virtual std::any visitFunctionBlock(std::shared_ptr<FunctionBlockNode> tree);
        // === BlOCK PROCEDURE NODES ===

        // === FUNCTION CALL NODE ===
        virtual std::any visitFunction_call(std::shared_ptr<FunctionCallNode> tree);

    };
}
