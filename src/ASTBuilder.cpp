#include "ASTBuilder.h"
#include "AST.h"

namespace gazprea {
    std::any ASTBuilder::visitFile(GazpreaParser::FileContext *ctx) {
        AST *t = new AST();
        for ( auto statement : ctx->statement() ) {
            t->addChild(visit(statement));
        }
        return t;
    }

    std::any ASTBuilder::visitVardecl(GazpreaParser::VardeclContext *ctx) {
        AST *t = new AST(GazpreaParser::VAR_DECL);
        t->addChild(visit(ctx->type()));
        t->addChild(new AST(ctx->ID()->getSymbol()));
        t->addChild(visit(ctx->expression()));
        return t;
    }

    std::any ASTBuilder::visitAssign(GazpreaParser::AssignContext *ctx) {
        AST *t = new AST(GazpreaParser::ASSIGN);
        t->addChild(new AST(ctx->ID()->getSymbol()));
        t->addChild(visit(ctx->expression()));
        return t;
    }

    std::any ASTBuilder::visitCond(GazpreaParser::CondContext *ctx) {
        AST *t = new AST(GazpreaParser::CONDITIONAL);
        t->addChild(visit(ctx->expression()));
        for (auto statement: ctx->statement()) {
            t->addChild(visit(statement));
        }
        return t;
    }

    std::any ASTBuilder::visitLoop(GazpreaParser::LoopContext *ctx) {
        AST *t = new AST(GazpreaParser::LOOP);
        t->addChild(visit(ctx->expression()));
        for (auto statement: ctx->statement()) {
            t->addChild(visit(statement));
        }
        return t;
    }

    std::any ASTBuilder::visitPrint(GazpreaParser::PrintContext *ctx) {
        AST *t = new AST(GazpreaParser::PRINT);
        t->addChild(visit(ctx->expression()));
        return t;
    }

    std::any ASTBuilder::visitParen(GazpreaParser::ParenContext *ctx) {
        AST *t = new AST(GazpreaParser::PARENTHESES);
        t->addChild(visit(ctx->expr()));
        return t;
    }

    std::any ASTBuilder::visitIndex(GazpreaParser::IndexContext *ctx) {
        AST *t = new AST(GazpreaParser::INDEX);
        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));
        return t;
    }

    std::any ASTBuilder::visitRange(GazpreaParser::RangeContext *ctx) {
        AST *t = new AST(GazpreaParser::RANGE);
        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));
        return t;
    }

    std::any ASTBuilder::visitGenerator(GazpreaParser::GeneratorContext *ctx) {
        AST *t = new AST(GazpreaParser::GENERATOR);
        t->addChild(new AST(ctx->ID()->getSymbol()));
        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));
        return t;
    }

    std::any ASTBuilder::visitFilter(GazpreaParser::FilterContext *ctx) {
        AST *t = new AST(GazpreaParser::FILTER);
        t->addChild(new AST(ctx->ID()->getSymbol()));
        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));
        return t;
    }

    std::any ASTBuilder::visitMath(GazpreaParser::MathContext *ctx) {
        AST *t = nullptr;
        if (ctx->op->getType() == GazpreaParser::MULT) {
            t = new AST(GazpreaParser::MULT);
        }
        else if (ctx->op->getType() == GazpreaParser::DIV){
            t = new AST(GazpreaParser::DIV);
        }
        else if (ctx->op->getType() == GazpreaParser::ADD){
            t = new AST(GazpreaParser::ADD);
        }
        else
            t = new AST(GazpreaParser::SUB);
        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));
        return t;
    }

    std::any ASTBuilder::visitCmp(GazpreaParser::CmpContext *ctx) {
        AST *t = nullptr;
        if (ctx->op->getType() == GazpreaParser::LT) {
            t = new AST(GazpreaParser::LT);
        }
        else if (ctx->op->getType() == GazpreaParser::GT){
            t = new AST(GazpreaParser::GT);
        }
        else if (ctx->op->getType() == GazpreaParser::EQ){
            t = new AST(GazpreaParser::EQ);
        }
        else
            t = new AST(GazpreaParser::NEQ);
        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));
        return t;
    }

    std::any ASTBuilder::visitLiteralID(GazpreaParser::LiteralIDContext *ctx) {
        return new AST(ctx->ID()->getSymbol());
    }

    std::any ASTBuilder::visitLiteralInt(GazpreaParser::LiteralIntContext *ctx) {
        return new AST(ctx->INT()->getSymbol());
    }

    std::any ASTBuilder::visitType(GazpreaParser::TypeContext *ctx) {
        return new AST(ctx->getStart());
    }

    std::any ASTBuilder::visitExpression(GazpreaParser::ExpressionContext *ctx) {
        AST *t = new AST(GazpreaParser::EXPRESSION);
        t->addChild(visit(ctx->expr()));
        return t;
    }
}
