#include "ASTBuilder.h"
#include "ASTNode/ArgNode.h"
#include "ASTNode/Method/FunctionNode.h"
#include "ASTNode/Loop/PredicatedLoopNode.h"
#include <memory>


#define DEBUG

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

    std::any ASTBuilder::visitTypedef(GazpreaParser::TypedefContext *ctx) {
#ifdef DEBUG
        std::cout << "visitTypedef " << ctx->ID()->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
        std::shared_ptr<ASTNode> t = std::make_shared<TypeDefNode>(ctx->getStart()->getLine(), sym);

        t->addChild(visit(ctx->type()));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitBaseType(GazpreaParser::BaseTypeContext *ctx) {
#ifdef DEBUG
        std::cout << "visit Type" << ctx->typeString->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->typeString->getText());
        std::shared_ptr<TypeNode> t = std::make_shared<TypeNode>(ctx->getStart()->getLine(), sym);

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    // STREAMS
    std::any ASTBuilder::visitStream(GazpreaParser::StreamContext *ctx) {
#ifdef DEBUG
        std::cout << "visitOutputStream" << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<StreamOut>(ctx->getStart()->getLine());

        t->addChild(visit(ctx->expression()));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitIdentity(GazpreaParser::IdentityContext *ctx) {
#ifdef DEBUG
        std::cout << "visitIdentity " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<NullNode>(ctx->getStart()->getLine());

        return t;
    }

    std::any ASTBuilder::visitNull(GazpreaParser::NullContext *ctx) {
#ifdef DEBUG
        std::cout << "visitNull " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<NullNode>(ctx->getStart()->getLine());

        return t;
    }


    // ============ LITERALS ===========

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

    std::any ASTBuilder::visitLiteralReal(GazpreaParser::LiteralRealContext *ctx) {
#ifdef DEBUG
        std::cout << "visitReal " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<RealNode>(ctx->getStart()->getLine(),std::stod(ctx->getText()));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitLiteralTuple(GazpreaParser::LiteralTupleContext *ctx) {
#ifdef DEBUG
        std::cout << "visitTuple" << ctx->getText() << std::endl;
#endif
        std::shared_ptr<TupleNode> t = std::make_shared<TupleNode>(ctx->getStart()->getLine());

        for (auto expr : ctx->expr()) {
          auto blockExpr = std::any_cast<std::shared_ptr<ASTNode>>(visit(expr));
          t->val.push_back(blockExpr);
        }

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitLiteralBoolean(GazpreaParser::LiteralBooleanContext *ctx) {
#ifdef DEBUG
        std::cout << "visitBool" << ctx->getText() << std::endl;
#endif
        std::shared_ptr<BoolNode> t = std::make_shared<BoolNode>(ctx->getStart()->getLine(), ctx->getText() == "true");

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitLiteralCharacter(GazpreaParser::LiteralCharacterContext *ctx) {
#ifdef DEBUG
        std::cout << "visitCharacter" << ctx->getText() << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<CharNode>(ctx->getStart()->getLine(),ctx->getText()[0]);

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitMath(GazpreaParser::MathContext *ctx) {
#ifdef DEBUG
        std::cout << "visitMath, op = " << ctx->op->getText() << std::endl;
#endif
        std::shared_ptr<BinaryArithNode> t = std::make_shared<BinaryArithNode>(ctx->getStart()->getLine());

        switch (ctx->op->getType()) {
            case GazpreaParser::EXP:
                t->op = BINOP::EXP;
                break;
            case GazpreaParser::MULT:
                t->op = BINOP::MULT;
                break;
            case GazpreaParser::DIV:
                t->op = BINOP::DIV;
                break;
            case GazpreaParser::ADD:
                t->op = BINOP::ADD;
                break;
            case GazpreaParser::SUB:
                t->op = BINOP::SUB;
                break;
            case GazpreaParser::REM:
                t->op = BINOP::REM;
                break;
            case GazpreaParser::CONCAT:
                t->op = BINOP::CONCAT;
                break;
            case GazpreaParser::DOT_PRODUCT:
                t->op = BINOP::DOT_PROD;
                break;
            case GazpreaParser::RESERVED_XOR:
                t->op = BINOP::XOR;
                break;
            case GazpreaParser::RESERVED_AND:
                t->op = BINOP::AND;
                break;
            default:
                throw std::runtime_error("unknown arithmetic operator " + ctx->op->getText());
        }

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitCmp(GazpreaParser::CmpContext *ctx) {
#ifdef DEBUG
        std::cout << "visitCmp, op = " << ctx->op->getText() << std::endl;
#endif
        std::shared_ptr<BinaryCmpNode> t = std::make_shared<BinaryCmpNode>(ctx->getStart()->getLine());

        switch (ctx->op->getType()) {
            case GazpreaParser::LT:
                t->op = BINOP::LTHAN;
                break;
            case GazpreaParser::GT:
                t->op = BINOP::GTHAN;
                break;
            case GazpreaParser::LE:
                t->op = BINOP::LEQ;
                break;
            case GazpreaParser::GE:
                t->op = BINOP::GEQ;
                break;
            case GazpreaParser::EQ:
                t->op = BINOP::EQUAL;
                break;
            case GazpreaParser::NEQ:
                t->op = BINOP::NEQUAL;
                break;
            default:
                throw std::runtime_error("unknown comparison operator " + ctx->op->getText());
        }

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitUnary(GazpreaParser::UnaryContext *ctx) {
#ifdef DEBUG
        std::cout << "visitUnary, op = " << ctx->op->getText() << std::endl;
#endif
        std::shared_ptr<UnaryExpr> t;

        switch (ctx->op->getType()) {
          case GazpreaParser::ADD:
              t->op = UNARYOP::POSITIVE;
              break;
          case GazpreaParser::SUB:
              t->op = UNARYOP::NEGATE;
              break;
          case GazpreaParser::RESERVED_NOT:
              t->op = UNARYOP::NOT;
              break;
          default:
              throw std::runtime_error("unknown unary operator " + ctx->op->getText());
        }

        t->addChild(visit(ctx->expr()));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }



    std::any ASTBuilder::visitAssign(GazpreaParser::AssignContext *ctx) {
        std::shared_ptr<Symbol> identifierSymbol = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<AssignNode> t = std::make_shared<AssignNode>(ctx->getStart()->getLine(), identifierSymbol);

        t->addChild(visit(ctx->lvalue));
        t->addChild(visit(ctx->rvalue));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitCond(GazpreaParser::CondContext *ctx) {
#ifdef DEBUGTUPLE
        std::cout << "visitCond " << std::endl;
#endif
        std::shared_ptr<ConditionalNode> t = std::make_shared<ConditionalNode>(ctx->getStart()->getLine());

        for (auto condition : ctx->expression()) {
          t->conditions.push_back(std::any_cast<std::shared_ptr<ASTNode>>(visit(condition)));
        }

        for (auto body : ctx->block()) {
          t->bodies.push_back(std::any_cast<std::shared_ptr<ASTNode>>(visit(body)));
        }

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

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

    /**
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
    */

    std::any ASTBuilder::visitBlock(GazpreaParser::BlockContext *ctx) {
        auto blockNode = std::make_shared<BlockNode>(ctx->getStart()->getLine());

        for (auto statement : ctx->statement()) {
          std::shared_ptr<ASTNode> statementNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(statement));
          blockNode->statements.push_back(statementNode);
        }

        return std::dynamic_pointer_cast<ASTNode>(blockNode);
    }

    std::any ASTBuilder::visitPredicatedLoop(GazpreaParser::PredicatedLoopContext *ctx) {
    }

    std::any ASTBuilder::visitInfiniteLoop(GazpreaParser::InfiniteLoopContext *ctx) {
    }

    std::any ASTBuilder::visitPostPredicatedLoop(GazpreaParser::PostPredicatedLoopContext *ctx) {
    }

    std::any ASTBuilder::visitIteratorLoop(GazpreaParser::IteratorLoopContext *ctx) {

    }

    std::any ASTBuilder::visitProcedure(GazpreaParser::ProcedureContext *ctx) {
#ifdef DEBUG
        std::cout << "Visiting procedure definition." << std::endl;
#endif
      std:: cout << ctx->ID()->getText();
      auto procSymbol = std::make_shared<Symbol>(ctx->ID()->getText());
      auto procedureNode = std::make_shared<ProcedureNode>(ctx->getStart()->getLine(), procSymbol);

      for (auto arg : ctx->parameter()) {
        auto argResult = std::any_cast<std::shared_ptr<ASTNode>>(visit(arg));

        procedureNode->orderedArgs.push_back(argResult);
      }

      if (ctx->block()) {
        auto blockResult = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->block()));
        procedureNode->body = blockResult;
      }

      return std::dynamic_pointer_cast<ASTNode>(procedureNode);
    }

    std::any ASTBuilder::visitFunction(GazpreaParser::FunctionContext *ctx) {
#ifdef DEBUG
        std::cout << "Visiting function definition." << std::endl;
#endif
      auto funcSymbol = std::make_shared<Symbol>(ctx->ID()->getText());
      auto functionNode = std::make_shared<FunctionNode>(ctx->getStart()->getLine(), funcSymbol);
      for (auto arg : ctx->parameter()) {
        auto argResult = std::any_cast<std::shared_ptr<ArgNode>>(visit(arg));

        functionNode->orderedArgs.push_back(argResult);
      }

      if (ctx->block()) {
        auto blockResult = std::any_cast<std::shared_ptr<BlockNode>>(visit(ctx->block()));
        functionNode->body = blockResult;
      }

      return std::dynamic_pointer_cast<ASTNode>(functionNode);
    }

    std::any ASTBuilder::visitParameter(GazpreaParser::ParameterContext *ctx) {
#ifdef DEBUG
        std::cout << "Visiting parameters" << std::endl;
#endif
      auto argNode = std::make_shared<ArgNode>(ctx->getStart()->getLine());


      std::shared_ptr<Symbol> identifierSymbol = std::make_shared<Symbol>(ctx->ID()->getText());

      std::shared_ptr<ASTNode> typeNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->type()));

      argNode->idSym = identifierSymbol;
      argNode->type = typeNode;

      return std::dynamic_pointer_cast<ASTNode>(argNode);
    }
}
