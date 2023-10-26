#include "ASTBuilder.h"
#include "AST.h"

#define DEBUG

namespace gazprea {
    std::any ASTBuilder::visitFile(GazpreaParser::FileContext *ctx) {
#ifdef DEBUG
        std::cout << "INIT BUILDER: VISITING FILE" << std::endl;
#endif

        AST *t = new AST();
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
        Symbol* sym = new Symbol(ctx->ID()->getSymbol()->getText());
        DeclNode *t = new DeclNode(GazpreaParser::VAR_DECL, sym);
        t->expr = std::any_cast<ExprAST*>(visit(ctx->expression()));
        t->type = std::any_cast<TypeNode*>(visit(ctx->type()));
        std::cout << "Returning vardecl" << std::endl;
        AST* ret = t;

        return ret;
    }

    std::any ASTBuilder::visitAssign(GazpreaParser::AssignContext *ctx) {
#ifdef DEBUG
        std::cout << "visitAssign " << ctx->getStart()->getType() << ": "
                  << ctx->ID()->getText() << std::endl;
#endif
        Symbol* sym = new Symbol(ctx->ID()->getSymbol()->getText());
        AssignNode *t = new AssignNode(GazpreaParser::VAR_DECL, sym);
        t->expr = std::any_cast<ExprAST*>(visit(ctx->expression()));

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
        RangeVecNode *t = new RangeVecNode(GazpreaParser::RANGE);
        t->left = std::any_cast<ExprAST*>(ctx->expr(0));
        t->right = std::any_cast<ExprAST*>(ctx->expr(1));
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
        // TODO remove token - we don't need it here!
        BinaryArithNode *t = new BinaryArithNode(ctx->op->getType());

        BINOP op;
        switch (ctx->op->getType()) {
            case GazpreaParser::MULT:
                op = BINOP::MULT;
                break;
            case GazpreaParser::DIV:
                op = BINOP::DIV;
                break;
            case GazpreaParser::ADD:
                op = BINOP::ADD;
            case GazpreaParser::SUB:
                op = BINOP::SUB;
        }
        t->left = std::any_cast<ExprAST*>(visit(ctx->expr(0)));
        t->right = std::any_cast<ExprAST*>(visit(ctx->expr(1)));
        t->op = op;
        return t;
    }

    std::any ASTBuilder::visitCmp(GazpreaParser::CmpContext *ctx) {
        BinaryCmpNode *t = new BinaryCmpNode(ctx->op->getType());

        BINOP op;
        switch (ctx->op->getType()) {
            case GazpreaParser::MULT:
                op = BINOP::MULT;
                break;
            case GazpreaParser::DIV:
                op = BINOP::DIV;
                break;
            case GazpreaParser::ADD:
                op = BINOP::ADD;
            case GazpreaParser::SUB:
                op = BINOP::SUB;
        }
        t->left = std::any_cast<ExprAST*>(visit(ctx->expr(0)));
        t->right = std::any_cast<ExprAST*>(visit(ctx->expr(1)));
        t->op = op;
        return t;
    }

    std::any ASTBuilder::visitLiteralID(GazpreaParser::LiteralIDContext *ctx) {
#ifdef DEBUG
        std::cout << "visitID " << ctx->ID()->getSymbol()->getText() << std::endl;
#endif
        Symbol* sym = new Symbol(ctx->ID()->getSymbol()->getText());
        return new IDNode(ctx->ID()->getSymbol(), sym);
    }

    std::any ASTBuilder::visitLiteralInt(GazpreaParser::LiteralIntContext *ctx) {
#ifdef DEBUG
        std::cout << "visitInt " << ctx->getText() << std::endl;
#endif
        ExprAST* t = new IntNode(GazpreaParser::INT,std::stoi(ctx->getText()));
        return t;
//        ExprAST* ret = new IntNode(GazpreaParser::INT,std::stoi(ctx->getText()));
//        std::cout << "INT: casting" << std::endl;
//        auto retCasted = std::any_cast<ExprAST*>(ret);
//        std::cout << "INT: done casting" << std::endl;
//        return retCasted;
    }

    std::any ASTBuilder::visitType(GazpreaParser::TypeContext *ctx) {
#ifdef DEBUG
        std::cout << "visitType " << ctx->getText() << std::endl;
#endif
        Symbol* sym = new Symbol(ctx->getText());
        return new TypeNode(ctx->getStart(), sym);
    }

    std::any ASTBuilder::visitExpression(GazpreaParser::ExpressionContext *ctx) {
#ifdef DEBUG
        std::cout << "visitExpression (parent) " << ctx->getText() << std::endl;
#endif
        // just return the inner expression
        // the parent expression is just to help the grammar, so it's not needed here
        return std::any_cast<ExprAST*>(visit(ctx->expr()));
    }
}
