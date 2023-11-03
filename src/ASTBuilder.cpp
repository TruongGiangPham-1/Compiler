#include "ASTBuilder.h"
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

//#define DEBUG

namespace gazprea {
    std::any ASTBuilder::visitFile(GazpreaParser::FileContext *ctx) {
#ifdef DEBUG
        std::cout << "INIT BUILDER: VISITING FILE" << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>();
        int size = ctx->children.size();
        int c = 0;
        for ( auto statement : ctx->children) {
            c++;
            if (c == size) break;
            t->addChild(visit(statement));
        }
        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    //std::any ASTBuilder::visitVardecl(GazpreaParser::VardeclContext *ctx) {
    //#ifdef DEBUG
    //    std::cout << "visitVarDecl type " << ctx->getStart()->getType() << ": "
    //              << ctx->ID()->getText() << std::endl;
    //#endif
    //    std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
    //    std::shared_ptr<ASTNode> t = std::make_shared<DeclNode>(ctx->getStart()->getLine(), sym);

    //    t->addChild(visit(ctx->type()));
    //    t->addChild(visit(ctx->expression()));

    //    return std::dynamic_pointer_cast<ASTNode>(t);
    //}

    std::any ASTBuilder::visitAssign(GazpreaParser::AssignContext *ctx) {
//#ifdef DEBUG
//        std::cout << "visitAssign " << ctx->getStart()->getType() << ": "
//                  << ctx->ID()->getText() << std::endl;
//#endif
//        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
//        std::shared_ptr<AssignNode> t = std::make_shared<AssignNode>(ctx->getStart()->getLine(), sym);
//
//        t->addChild(visit(ctx->expression()));

//        return std::dynamic_pointer_cast<ASTNode>(t);
        return nullptr;
    }

    std::any ASTBuilder::visitCond(GazpreaParser::CondContext *ctx) {
#ifdef DEBUG
        std::cout << "visitCond " << std::endl;
#endif
        std::shared_ptr<ConditionalNode> t = std::make_shared<ConditionalNode>(ctx->getStart()->getLine());
        //t->condition = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));
        //for (auto statement: ctx->statement()) {
        //    t->addChild(visit(statement));
        //}

        //return std::dynamic_pointer_cast<ASTNode>(t);
        return nullptr;
    }

    //std::any ASTBuilder::visitLoop(GazpreaParser::LoopContext *ctx) {
//#ifdef DEBUG
    //    std::cout << "visitLoop" << std::endl;
//#end//if
    //    std::shared_ptr<LoopNode> t = std::make_shared<LoopNode>(ctx->getStart()->getLine());
    //    t->condition = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));

    //    for (auto statement: ctx->statement()) {
    //        t->addChild(visit(statement));
    //    }

    //    // dynamically casting upward to an ASTNode
    //    return std::dynamic_pointer_cast<ASTNode>(t);
    //}

    //std::any ASTBuilder::visitPrint(GazpreaParser::PrintContext *ctx) {
//#ifdef DEBUG
    //    std::cout << "visitPrint" << std::endl;
//#endif
    //    std::shared_ptr<ASTNode> t = std::make_shared<PrintNode>(ctx->getStart()->getLine());

    //    t->addChild(visit(ctx->expression()));

    //    return std::dynamic_pointer_cast<ASTNode>(t);
    //}

//    std::any ASTBuilder::visitParen(GazpreaParser::ParenContext *ctx) {
//#ifdef DEBUG
//        std::cout << "visitParen" << std::endl;
//#endif
//        // no need to make an AST node for this rule
//        return visit(ctx->expr());
//    }

    std::any ASTBuilder::visitIndex(GazpreaParser::IndexContext *ctx) {
#ifdef DEBUG
        std::cout << "visitIndex" << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<IndexNode>(ctx->getStart()->getLine());

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitRange(GazpreaParser::RangeContext *ctx) {
#ifdef DEBUG
        std::cout << "visitRange" << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<RangeVecNode>(ctx->getStart()->getLine());

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitGenerator(GazpreaParser::GeneratorContext *ctx) {
#ifdef DEBUG
        std::cout << "visitGenerator" << std::endl;
#endif
        std::string domainVar = ctx->ID(0)->getSymbol()->getText();
        std::shared_ptr<ASTNode> t = std::make_shared<GeneratorNode>(domainVar, ctx->getStart()->getLine());

        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitFilter(GazpreaParser::FilterContext *ctx) {
#ifdef DEBUG
        std::cout << "visitFilter" << std::endl;
#endif
        std::string domainVar = ctx->ID()->getSymbol()->getText();
        std::shared_ptr<ASTNode> t = std::make_shared<FilterNode>(domainVar, ctx->getStart()->getLine());

        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitMath(GazpreaParser::MathContext *ctx) {
#ifdef DEBUG
        std::cout << "visitMath, op = " << ctx->op->getText() << std::endl;
#endif
        std::shared_ptr<ArithNode> t = std::make_shared<ArithNode>(ctx->getStart()->getLine());

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
#ifdef DEBUG
        std::cout << "visitCmp, op = " << ctx->op->getText() << std::endl;
#endif
        std::shared_ptr<CmpNode> t = std::make_shared<CmpNode>(ctx->getStart()->getLine());

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
        std::shared_ptr<ASTNode> t = std::make_shared<IDNode>(ctx->getStart()->getLine(), sym);

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitLiteralInt(GazpreaParser::LiteralIntContext *ctx) {
#ifdef DEBUG
        std::cout << "visitInt " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<IntNode>(ctx->getStart()->getLine(),std::stoi(ctx->getText()));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitType(GazpreaParser::TypeContext *ctx) {
#ifdef DEBUG
        std::cout << "visitType " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<ASTNode> t = std::make_shared<TypeNode>(ctx->getStart()->getLine(), sym);

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitExpression(GazpreaParser::ExpressionContext *ctx) {
#ifdef DEBUG
        std::cout << "visitExpression (parent) " << ctx->getText() << std::endl;
#endif
        // just return the inner expression
        // the parent expression is just to help the grammar, so it's not needed here
        return visit(ctx->expr());
    }

    std::any ASTBuilder::visitFunctionSingle(GazpreaParser::FunctionSingleContext *ctx) {
        std::cout << "visiting function Single\n";
        // ctx->ID(0) is always the function name, kind of mystery indexing
        std::shared_ptr<Symbol> funcNameSym = std::make_shared<Symbol>(ctx->ID(0)->getSymbol()->getText());
        std::shared_ptr<FunctionSingleNode> t = std::make_shared<FunctionSingleNode>(ctx->getStart()->getLine(), funcNameSym);

        auto typesArray = ctx->type();
        auto argIDArray = ctx->ID();
        // TODO: add the retType node(doesnt work yet without visitType)
        //t->addChild(visit(typesArray[typesArray.size() - 1])); // the last type is always the return type?

        // iterate thru all the orderedArg
        // ctx->ID() is an array of arguments id, skip ID(0) because thats the function name
        // type of the argument ID(i) is at ctx->type(i - 1)
        for (long unsigned int i = 1; i < argIDArray.size(); i++) {
            // TODO: cant visit type yet since its not yet done
            //auto argTypeNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(typesArray[i - 1]));
            auto  sym = std::make_shared<Symbol>(argIDArray[i]->getSymbol()->getText());
            auto argIDnode = std::make_shared<IDNode>(argIDArray[i]->getSymbol()->getLine(), sym);
            // TODO:  set the type of the ID node once visit Type is implemented
            //argIDnode->type = std::any_cast<std::shared_ptr<Type>>(visit(typesArray[i - 1]));

            t->orderedArgs.push_back(argIDnode);

        }
        t->addChild(visit(ctx->expression()));
        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitFunctionBlock(GazpreaParser::FunctionBlockContext *ctx) {
        std::cout << "visiting function block\n";
        // ctx->ID(0) is always the function name, kind of mystery indexing
        std::shared_ptr<Symbol> funcNameSym = std::make_shared<Symbol>(ctx->ID(0)->getSymbol()->getText());
        std::shared_ptr<FunctionBlockNode> t = std::make_shared<FunctionBlockNode>(ctx->getStart()->getLine(), funcNameSym);

        auto typesArray = ctx->type();
        auto argIDArray = ctx->ID();
        // TODO: add the retType node(doesnt work yet without type walker)
        //t->addChild(visit(typesArray[typesArray.size() - 1])); // the last type is always the return type?

        // iterate thru all the orderedArg
        // ctx->ID() is an array of arguments id, skip ID(0) because thats the function name
        // type of the argument ID(i) is at ctx->type(i - 1)
        for (long unsigned int i = 1; i < argIDArray.size(); i++) {
            // TODO: cant visit type yet since its not yet done
            //auto argTypeNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(typesArray[i - 1]));
            auto  sym = std::make_shared<Symbol>(argIDArray[i]->getSymbol()->getText());
            auto argIDnode = std::make_shared<IDNode>(argIDArray[i]->getSymbol()->getLine(), sym);
            // TODO:  set the type of the ID node once visit Type is implemented
            //argIDnode->type = std::any_cast<std::shared_ptr<Type>>(visit(typesArray[i - 1]));

            t->orderedArgs.push_back(argIDnode);

        }
        // TODO: add block node when visitBlock is implemented
        //t->addChild(visit(ctx->block()));
        return std::dynamic_pointer_cast<ASTNode>(t);

    }

    // func decl node, holds retType,
    std::any ASTBuilder::visitFunctionForward(GazpreaParser::FunctionForwardContext *ctx) {
        std::cout << "visiting function forwward\n";
        // ctx->ID(0) is always the function name, kind of mystery indexing
        std::shared_ptr<Symbol> funcNameSym = std::make_shared<Symbol>(ctx->ID(0)->getSymbol()->getText());
        std::shared_ptr<FunctionForwardNode> t = std::make_shared<FunctionForwardNode>(ctx->getStart()->getLine(), funcNameSym);

        auto typesArray = ctx->type();
        auto argIDArray = ctx->ID();
        // TODO: add the retType node(doesnt work yet without type walker)
        //t->addChild(visit(typesArray[typesArray.size() - 1])); // the last type is always the return type?



        // iterate thru all the orderedArg
        // ctx->ID() is an array of arguments id, skip ID(0) because thats the function name
        // type of the argument ID(i) is at ctx->type(i - 1)
        for (long unsigned int i = 1; i < argIDArray.size(); i++) {
            // TODO: cant visit type yet since its not yet done
            //auto argTypeNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(typesArray[i - 1]));
            auto  sym = std::make_shared<Symbol>(argIDArray[i]->getSymbol()->getText());
            auto argIDnode = std::make_shared<IDNode>(argIDArray[i]->getSymbol()->getLine(), sym);
            // TODO:  set the type of the ID node once visit Type is implemented
            //argIDnode->type = std::any_cast<std::shared_ptr<Type>>(visit(typesArray[i - 1]));

            t->orderedArgs.push_back(argIDnode);

        }
        // doesnt have the block expression

        return std::dynamic_pointer_cast<ASTNode>(t);
    }
}
