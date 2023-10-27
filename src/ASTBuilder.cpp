#include "ASTBuilder.h"
#include "AST.h"

#define DEBUG

namespace gazprea {
    std::any ASTBuilder::visitFile(GazpreaParser::FileContext *ctx) {
#ifdef DEBUG
        std::cout << "INIT BUILDER: VISITING FILE" << std::endl;
#endif
        std::shared_ptr<AST> t = std::make_shared<AST>();
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
        std::shared_ptr<DeclNode> t = std::make_shared<DeclNode>(GazpreaParser::VAR_DECL, sym);

        t->type = std::any_cast<std::shared_ptr<TypeNode>>(visit(ctx->type()));
        std::cout << "About to visit inner expr of vardecl" << std::endl;
        t->expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expression()));
        std::cout << "Returning vardecl" << std::endl;

        // go up the hierarchy, since the parent (visitFile) expects an AST
        // without this, we'll get a bad any cast err
        return std::static_pointer_cast<AST>(t);
    }

    std::any ASTBuilder::visitAssign(GazpreaParser::AssignContext *ctx) {
#ifdef DEBUG
        std::cout << "visitAssign " << ctx->getStart()->getType() << ": "
                  << ctx->ID()->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
        std::shared_ptr<AssignNode> t = std::make_shared<AssignNode>(GazpreaParser::ASSIGN, sym);

        t->expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expression()));

        // go up the hierarchy, since the parent (visitFile) expects an AST
        return std::static_pointer_cast<AST>(t);
    }

    std::any ASTBuilder::visitCond(GazpreaParser::CondContext *ctx) {
        std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::CONDITIONAL);
        t->addChild(visit(ctx->expression()));
        for (auto statement: ctx->statement()) {
            t->addChild(visit(statement));
        }
        return t;
    }

    std::any ASTBuilder::visitLoop(GazpreaParser::LoopContext *ctx) {
        std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::CONDITIONAL);
        t->addChild(visit(ctx->expression()));
        for (auto statement: ctx->statement()) {
            t->addChild(visit(statement));
        }
        return t;
    }

    std::any ASTBuilder::visitPrint(GazpreaParser::PrintContext *ctx) {
        std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::PRINT);
        t->addChild(visit(ctx->expression()));
        return t;
    }

    std::any ASTBuilder::visitParen(GazpreaParser::ParenContext *ctx) {
        std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::PARENTHESES);
        t->addChild(visit(ctx->expr()));
        return t;
    }

    std::any ASTBuilder::visitIndex(GazpreaParser::IndexContext *ctx) {
        std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::INDEX);
        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));
        return t;
    }

    std::any ASTBuilder::visitRange(GazpreaParser::RangeContext *ctx) {
        std::shared_ptr<RangeVecNode> t = std::make_shared<RangeVecNode>(GazpreaParser::RANGE);
        t->left = std::any_cast<std::shared_ptr<ExprAST>>(ctx->expr(0));
        t->right = std::any_cast<std::shared_ptr<ExprAST>>(ctx->expr(1));

        // parents of these nodes expect an ExprAST. Without the cast, we'll get a bad_any cast
        return std::static_pointer_cast<ExprAST>(t);
    }

    std::any ASTBuilder::visitGenerator(GazpreaParser::GeneratorContext *ctx) {
        std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::GENERATOR);
        t->addChild(new AST(ctx->ID()->getSymbol()));
        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));
        return t;
    }

    std::any ASTBuilder::visitFilter(GazpreaParser::FilterContext *ctx) {
        std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::FILTER);
        t->addChild(new AST(ctx->ID()->getSymbol()));
        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));
        return t;
    }

    std::any ASTBuilder::visitMath(GazpreaParser::MathContext *ctx) {
        // TODO remove token in constructor - we don't need it here
        std::shared_ptr<BinaryArithNode> t = std::make_shared<BinaryArithNode>(ctx->op->getType());

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
        t->left = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
        t->right = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
        t->op = op;

        // parents of these nodes expect an ExprAST. Without the cast, we'll get a bad_any cast
        return std::static_pointer_cast<ExprAST>(t);
    }

    std::any ASTBuilder::visitCmp(GazpreaParser::CmpContext *ctx) {
        // TODO remove token in constructor - we don't need it here
        std::shared_ptr<BinaryCmpNode> t = std::make_shared<BinaryCmpNode>(ctx->op->getType());

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
        t->left = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
        t->right = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
        t->op = op;

        // parents of these nodes expect an ExprAST. Without the cast, we'll get a bad_any cast
        return std::static_pointer_cast<ExprAST>(t);
    }

    std::any ASTBuilder::visitLiteralID(GazpreaParser::LiteralIDContext *ctx) {
#ifdef DEBUG
        std::cout << "visitID " << ctx->ID()->getSymbol()->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
        std::shared_ptr<IDNode> t = std::make_shared<IDNode>(ctx->ID()->getSymbol(), sym);

        // parents of these nodes expect an ExprAST. Without the cast, we'll get a bad_any cast
//        return std::static_pointer_cast<ExprAST>(t);
        return t;
    }

    std::any ASTBuilder::visitLiteralInt(GazpreaParser::LiteralIntContext *ctx) {
#ifdef DEBUG
        std::cout << "visitInt " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<IntNode> t = std::make_shared<IntNode>(GazpreaParser::INT,std::stoi(ctx->getText()));

        // parents of these nodes expect an ExprAST. Without the cast, we'll get a bad_any cast
        return std::static_pointer_cast<ExprAST>(t);
    }

    std::any ASTBuilder::visitType(GazpreaParser::TypeContext *ctx) {
#ifdef DEBUG
        std::cout << "visitType " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<TypeNode> t = std::make_shared<TypeNode>(ctx->getStart(), sym);

        // parents of the TypeNode expect no parent node. No casting required.
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
