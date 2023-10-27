#include "ASTBuilder.h"
#include "ASTNode/ASTNode.h"

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
        std::shared_ptr<ASTNode> t = std::make_shared<DeclNode>(GazpreaParser::VAR_DECL, sym);

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
        std::shared_ptr<AssignNode> t = std::make_shared<AssignNode>(GazpreaParser::ASSIGN, sym);

        t->addChild(visit(ctx->expression()));

        return t;
    }

    std::any ASTBuilder::visitCond(GazpreaParser::CondContext *ctx) {
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>(GazpreaParser::CONDITIONAL);
        t->addChild(visit(ctx->expression()));
        for (auto statement: ctx->statement()) {
            t->addChild(visit(statement));
        }
        return t;
    }

    std::any ASTBuilder::visitLoop(GazpreaParser::LoopContext *ctx) {
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>(GazpreaParser::CONDITIONAL);

        t->addChild(visit(ctx->expression()));
        for (auto statement: ctx->statement()) {
            t->addChild(visit(statement));
        }

        return t;
    }

    std::any ASTBuilder::visitPrint(GazpreaParser::PrintContext *ctx) {
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>(GazpreaParser::PRINT);

        t->addChild(visit(ctx->expression()));

        return t;
    }

    std::any ASTBuilder::visitParen(GazpreaParser::ParenContext *ctx) {
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>(GazpreaParser::PARENTHESES);

        t->addChild(visit(ctx->expr()));

        return t;
    }

    std::any ASTBuilder::visitIndex(GazpreaParser::IndexContext *ctx) {
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>(GazpreaParser::INDEX);

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        return t;
    }

    std::any ASTBuilder::visitRange(GazpreaParser::RangeContext *ctx) {
        std::shared_ptr<ASTNode> t = std::make_shared<RangeVecNode>(GazpreaParser::RANGE);

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        return std::static_pointer_cast<ExprNode>(t);
    }

    std::any ASTBuilder::visitGenerator(GazpreaParser::GeneratorContext *ctx) {
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>(GazpreaParser::GENERATOR);

        t->addChild(new ASTNode(ctx->ID()->getSymbol()));
        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));

        return t;
    }

    std::any ASTBuilder::visitFilter(GazpreaParser::FilterContext *ctx) {
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>(GazpreaParser::FILTER);

        t->addChild(new ASTNode(ctx->ID()->getSymbol()));
        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));

        return t;
    }

    std::any ASTBuilder::visitMath(GazpreaParser::MathContext *ctx) {
        std::shared_ptr<BinaryArithNode> t = std::make_shared<BinaryArithNode>(ctx->op->getType());

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
        std::shared_ptr<BinaryCmpNode> t = std::make_shared<BinaryCmpNode>(ctx->op->getType());

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
        std::shared_ptr<ASTNode> t = std::make_shared<IDNode>(ctx->ID()->getSymbol(), sym);

        return t;
    }

    std::any ASTBuilder::visitLiteralInt(GazpreaParser::LiteralIntContext *ctx) {
#ifdef DEBUG
        std::cout << "visitInt " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<IntNode>(GazpreaParser::INT,std::stoi(ctx->getText()));

        return t;
    }

    std::any ASTBuilder::visitType(GazpreaParser::TypeContext *ctx) {
#ifdef DEBUG
        std::cout << "visitType " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<ASTNode> t = std::make_shared<TypeNode>(ctx->getStart(), sym);

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
