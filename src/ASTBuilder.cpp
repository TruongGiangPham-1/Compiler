#pragma once

#include "ASTBuilder.h"
#include "ASTNode/ASTNode.h"
#include "ASTNode/AssignNode.h"
#include "ASTNode/TypeNode.h"
#include "ASTNode/DeclNode.h"
#include "ASTNode/Expr/IDNode.h"
#include "ASTNode/Expr/IntNode.h"
#include "ASTNode/Expr/Binary/BinaryExpr.h"
#include "ASTNode/Expr/Vector/RangeVecNode.h"
#include "ASTNode/Expr/Vector/GeneratorNode.h"
#include "ASTNode/Expr/Vector/FilterNode.h"
#include "ASTNode/Expr/Unary/UnaryExpr.h"
#include "ASTNode/Block/LoopNode.h"
#include "ASTNode/Block/ConditionalNode.h"

#define DEBUG

namespace gazprea {
    std::any ASTBuilder::visitFile(GazpreaParser::FileContext *ctx) {
#ifdef DEBUG
        std::cout << "INIT BUILDER: VISITING FILE" << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>();
        for ( auto statement : ctx->statement() ) {

            t->addChild(visit(statement));
        }
        return t;
    }

    std::any ASTBuilder::visitVardecl(GazpreaParser::VardeclContext *ctx) {
#ifdef DEBUG
        std::cout << "visitVarDecl type " << ctx->getStart()->getType() << ": "
                  << ctx->ID()->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
        std::shared_ptr<ASTNode> t = std::make_shared<DeclNode>(GazpreaParser::VAR_DECL, ctx->getStart()->getLine(), sym);

        t->addChild(visit(ctx->type()));
        t->addChild(visit(ctx->expression()));

        return t;
    }

    std::any ASTBuilder::visitAssign(GazpreaParser::AssignContext *ctx) {
#ifdef DEBUG
        std::cout << "visitAssign " << ctx->getStart()->getType() << ": "
                  << ctx->ID()->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
        std::shared_ptr<AssignNode> t = std::make_shared<AssignNode>(GazpreaParser::ASSIGN, ctx->getStart()->getLine(), sym);

        t->addChild(visit(ctx->expression()));

        return t;
    }

    std::any ASTBuilder::visitCond(GazpreaParser::CondContext *ctx) {
#ifdef DEBUG
        std::cout << "visitCond " << std::endl;
#endif
        std::shared_ptr<ConditionalNode> t = std::make_shared<ConditionalNode>(GazpreaParser::CONDITIONAL, ctx->getStart()->getLine());
        t->condition = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));
        for (auto statement: ctx->statement()) {
            t->addChild(visit(statement));
        }

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitLoop(GazpreaParser::LoopContext *ctx) {
#ifdef DEBUG
        std::cout << "visitLoop" << std::endl;
#endif
        std::shared_ptr<LoopNode> t = std::make_shared<LoopNode>(GazpreaParser::LOOP, ctx->getStart()->getLine());
        t->condition = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));

        for (auto statement: ctx->statement()) {
            t->addChild(visit(statement));
        }

        // dynamically casting upward to an ASTNode
        return std::dynamic_pointer_cast<ASTNode>(t);;
    }

    std::any ASTBuilder::visitPrint(GazpreaParser::PrintContext *ctx) {
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>(GazpreaParser::PRINT, ctx->getStart()->getLine());

        t->addChild(visit(ctx->expression()));

        return t;
    }

    std::any ASTBuilder::visitParen(GazpreaParser::ParenContext *ctx) {
        // no need to make an AST node for this rule
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>(GazpreaParser::PARENTHESES, ctx->getStart()->getLine());

        t->addChild(visit(ctx->expr()));

        return t;
    }

    std::any ASTBuilder::visitIndex(GazpreaParser::IndexContext *ctx) {
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>(GazpreaParser::INDEX, ctx->getStart()->getLine());

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        return t;
    }

    std::any ASTBuilder::visitRange(GazpreaParser::RangeContext *ctx) {
#ifdef DEBUG
        std::cout << "visitRange" << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<RangeVecNode>(GazpreaParser::RANGE, ctx->getStart()->getLine());

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        return t;
    }

    std::any ASTBuilder::visitGenerator(GazpreaParser::GeneratorContext *ctx) {
#ifdef DEBUG
        std::cout << "visitGenerator" << std::endl;
#endif
        std::string domainVar = ctx->ID()->getSymbol()->getText();
        std::shared_ptr<ASTNode> t = std::make_shared<GeneratorNode>(GazpreaParser::GENERATOR, domainVar, ctx->getStart()->getLine());

        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));

        return t;
    }

    std::any ASTBuilder::visitFilter(GazpreaParser::FilterContext *ctx) {
#ifdef DEBUG
        std::cout << "visitFilter" << std::endl;
#endif
        std::string domainVar = ctx->ID()->getSymbol()->getText();
        std::shared_ptr<ASTNode> t = std::make_shared<FilterNode>(GazpreaParser::FILTER, domainVar, ctx->getStart()->getLine());

        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));

        return t;
    }

    std::any ASTBuilder::visitMath(GazpreaParser::MathContext *ctx) {
        std::shared_ptr<BinaryArithNode> t = std::make_shared<BinaryArithNode>(GazpreaParser::EXPRESSION, ctx->getStart()->getLine());

        switch (ctx->op->getType()) {
            case GazpreaParser::MULT:
                t->op = BINOP::MULT;
                break;
            case GazpreaParser::DIV:
                t->op = BINOP::DIV;
                break;
            case GazpreaParser::ADD:
                t->op = BINOP::ADD;
            case GazpreaParser::SUB:
                t->op = BINOP::SUB;
        }

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        // casting upward to an ASTNode
        // we want to use the .op attribute, so we don't want to cast it upward when initializing
        // like in the case of most other nodes
        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitCmp(GazpreaParser::CmpContext *ctx) {
        std::shared_ptr<BinaryCmpNode> t = std::make_shared<BinaryCmpNode>(GazpreaParser::EXPRESSION, ctx->getStart()->getLine());

        switch (ctx->op->getType()) {
            case GazpreaParser::MULT:
                t->op = BINOP::MULT;
                break;
            case GazpreaParser::DIV:
                t->op = BINOP::DIV;
                break;
            case GazpreaParser::ADD:
                t->op = BINOP::ADD;
            case GazpreaParser::SUB:
                t->op = BINOP::SUB;
        }

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        // casting upward to an ASTNode
        // we want to use the .op attribute, so we don't want to cast it upward when initializing
        // like in the case of most other nodes
        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitLiteralID(GazpreaParser::LiteralIDContext *ctx) {
#ifdef DEBUG
        std::cout << "visitID " << ctx->ID()->getSymbol()->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
        std::shared_ptr<ASTNode> t = std::make_shared<IDNode>(GazpreaParser::ID, ctx->getStart()->getLine(), sym);

        return t;
    }

    std::any ASTBuilder::visitLiteralInt(GazpreaParser::LiteralIntContext *ctx) {
#ifdef DEBUG
        std::cout << "visitInt " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<IntNode>(GazpreaParser::INT, ctx->getStart()->getLine(),std::stoi(ctx->getText()));

        return t;
    }

    std::any ASTBuilder::visitType(GazpreaParser::TypeContext *ctx) {
#ifdef DEBUG
        std::cout << "visitType " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<ASTNode> t = std::make_shared<TypeNode>(GazpreaParser::TYPE, ctx->getStart()->getLine(), sym);

        return t;
    }

    std::any ASTBuilder::visitExpression(GazpreaParser::ExpressionContext *ctx) {
#ifdef DEBUG
        std::cout << "visitExpression (parent) " << ctx->getText() << std::endl;
#endif
        // just return the inner expression
        // the parent expression is just to help the grammar, so it's not needed here
        return visit(ctx->expr());
    }
}
