#include "ASTBuilder.h"
#include "ASTNode/ArgNode.h"
#include "ASTNode/Expr/CastNode.h"
#include "ASTNode/Expr/StdInputNode.h"
#include "ASTNode/Expr/TupleIndexNode.h"
#include "ASTNode/Method/FunctionNode.h"
#include "ASTNode/Type/TypeNode.h"
#include <memory>

// #define DEBUG

namespace gazprea {
std::any ASTBuilder::visitFile(GazpreaParser::FileContext* ctx)
{
#ifdef DEBUG
    std::cout << "INIT BUILDER: VISITING FILE" << std::endl;
#endif
    std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>();
    int size = ctx->children.size();
    int c = 0;
    for (auto statement : ctx->children) {
        c++;
        if (c == size)
            break;
        t->addChild(visit(statement));
    }
    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitTypedef(GazpreaParser::TypedefContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitTypedef " << ctx->ID()->getText() << std::endl;
#endif
    std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
    std::shared_ptr<ASTNode> t = std::make_shared<TypeDefNode>(ctx->getStart()->getLine(), sym);

    t->addChild(visit(ctx->type()));

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitCast(GazpreaParser::CastContext* ctx)
{
#ifdef DEBUG
    std::cout << "visit Typecast" << std::endl;
#endif
    std::shared_ptr<CastNode> t = std::make_shared<CastNode>(ctx->getStart()->getLine());
    std::shared_ptr<ASTNode> toType = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->type()));
    std::shared_ptr<ASTNode> expr = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));

    t->addChild(toType);
    t->addChild(expr);

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitBaseType(GazpreaParser::BaseTypeContext* ctx)
{
#ifdef DEBUG
    std::cout << "visit Type" << ctx->typeString->getText() << std::endl;
#endif

    if (ctx->typeString->getText() == "string") {
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>("character[*]");

        std::shared_ptr<Symbol> baseSym = std::make_shared<Symbol>("character");
        std::shared_ptr<TypeNode> baseType = std::make_shared<TypeNode>(ctx->getStart()->getLine(), baseSym);
        auto innerType = std::dynamic_pointer_cast<ASTNode>(baseType);

        auto t = std::make_shared<VectorTypeNode>(ctx->getStart()->getLine(), sym, innerType);
        t->isString = true;
        return std::dynamic_pointer_cast<ASTNode>(t);
    } else {
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->typeString->getText());
        std::shared_ptr<TypeNode> t = std::make_shared<TypeNode>(ctx->getStart()->getLine(), sym);
        return std::dynamic_pointer_cast<ASTNode>(t);
    }
}

std::any ASTBuilder::visitVectorType(GazpreaParser::VectorTypeContext* ctx)
{
#ifdef DEBUG
    std::cout << "visit VectorType" << ctx->getText() << std::endl;
#endif
    std::shared_ptr<Symbol> sym = nullptr;
    std::shared_ptr<ASTNode> innerType = nullptr;
    std::shared_ptr<VectorTypeNode> t;
    if (ctx->type()->getText() == "string") {
        std::string typeText = std::string("character") + "[" + ctx->typeSize()->getText() + "]";
        sym = std::make_shared<Symbol>(typeText);
        std::shared_ptr<Symbol> baseSym = std::make_shared<Symbol>("character");
        std::shared_ptr<TypeNode> baseType = std::make_shared<TypeNode>(ctx->getStart()->getLine(), baseSym);
        innerType = std::dynamic_pointer_cast<ASTNode>(baseType);
        t = std::make_shared<VectorTypeNode>(ctx->getStart()->getLine(), sym, innerType);
        t->isString = true;
    } else {
        sym = std::make_shared<Symbol>(ctx->getText());
        innerType = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->type()));
        t = std::make_shared<VectorTypeNode>(ctx->getStart()->getLine(), sym, innerType);
    }

    if (ctx->typeSize()->expression()) {
        // size is specified
        auto size = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->typeSize()->expression()));
        t->size = size;
    }

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitMatrixType(GazpreaParser::MatrixTypeContext* ctx)
{
#ifdef DEBUG
    std::cout << "visit visitMatrixType" << ctx->getText() << std::endl;
#endif
    std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());

    // inner type
    auto innerType = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->type()));
    auto t = std::make_shared<MatrixTypeNode>(ctx->getStart()->getLine(), sym, innerType);

    if (ctx->typeSize(0)->expression()) {
        // left size
        auto size = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->typeSize(0)->expression()));
        t->sizeLeft = size;
    }

    if (ctx->typeSize(1)->expression()) {
        // right size
        auto size = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->typeSize(1)->expression()));
        t->sizeRight = size;
    }

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitTupleType(GazpreaParser::TupleTypeContext* ctx)
{
#ifdef DEBUG
    std::cout << "visit TupleType" << ctx->getText() << std::endl;
#endif
    std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
    auto t = std::make_shared<TupleTypeNode>(ctx->getStart()->getLine(), sym);

    for (auto tupleField : ctx->tupleTypeField()) {
        auto type = std::any_cast<std::shared_ptr<ASTNode>>(visit(tupleField->type()));
        if (tupleField->ID()) {
            // has name
            std::string id = tupleField->ID()->getText();
            t->innerTypes.push_back(std::make_pair(id, type));
        } else {
            // no name
            t->innerTypes.push_back(std::make_pair("", type));
        }
    }

    return std::dynamic_pointer_cast<ASTNode>(t);
}

// STREAMS
std::any ASTBuilder::visitStreamOut(GazpreaParser::StreamOutContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitOutputStream" << std::endl;
#endif
    std::shared_ptr<ASTNode> t = std::make_shared<StreamOut>(ctx->getStart()->getLine());

    t->addChild(visit(ctx->expression()));

    return t;
}

std::any ASTBuilder::visitStreamIn(GazpreaParser::StreamInContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitInputStream" << std::endl;
#endif
    std::shared_ptr<ASTNode> t = std::make_shared<StreamIn>(ctx->getStart()->getLine());

    t->addChild(visit(ctx->expression()));

    return t;
}

std::any ASTBuilder::visitStreamStateFunctionCall(GazpreaParser::StreamStateFunctionCallContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitStreamStateFunctionCall" << std::endl;
#endif
    // this is a regular function call but with the name "streamState"
    std::shared_ptr<CallNode> callNode = std::make_shared<CallNode>(ctx->getStart()->getLine());
    std::shared_ptr<Symbol> funcName = std::make_shared<Symbol>("stream_state");

    callNode->CallName = funcName;

    // add dummy StdInput node
    // this helps stream_state work with our current function call codegen
    std::shared_ptr<ASTNode> stdInput = std::make_shared<StdInputNode>(ctx->getStart()->getLine());
    callNode->addChild(stdInput);

    return std::dynamic_pointer_cast<ASTNode>(callNode);
}

std::any ASTBuilder::visitStreamStateProcedureCall(GazpreaParser::StreamStateProcedureCallContext* ctx)
{
    // a procedure call to streamState is a NoOp
    // thus, do nothing
    return std::make_shared<ASTNode>();
}

std::any ASTBuilder::visitIdentity(GazpreaParser::IdentityContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitIdentity " << ctx->getText() << std::endl;
#endif
    std::shared_ptr<ASTNode> t = std::make_shared<IdentityNode>(ctx->getStart()->getLine());

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitNull(GazpreaParser::NullContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitNull " << ctx->getText() << std::endl;
#endif
    std::shared_ptr<ASTNode> t = std::make_shared<NullNode>(ctx->getStart()->getLine());

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitParentheses(GazpreaParser::ParenthesesContext* ctx)
{
    return visit(ctx->expr());
}
// ============ LITERALS ===========

std::any ASTBuilder::visitLiteralID(GazpreaParser::LiteralIDContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitID " << ctx->ID()->getSymbol()->getText() << std::endl;
#endif
    std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
    std::shared_ptr<ASTNode> t = std::make_shared<IDNode>(ctx->getStart()->getLine(), sym);

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitLiteralInt(GazpreaParser::LiteralIntContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitInt " << ctx->getText() << std::endl;
#endif
    try {
        std::stoi(ctx->getText());
    } catch (std::out_of_range& e) {
        throw LiteralError(ctx->getStart()->getLine(), "integer literal out of range " + ctx->getText());
    }

    std::shared_ptr<ASTNode> t = std::make_shared<IntNode>(ctx->getStart()->getLine(), std::stoi(ctx->getText()));

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitLiteralReal(GazpreaParser::LiteralRealContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitReal " << ctx->getText() << std::endl;
#endif
    std::shared_ptr<ASTNode> t = std::make_shared<RealNode>(ctx->getStart()->getLine(), std::stod(ctx->getText()));

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitLiteralTuple(GazpreaParser::LiteralTupleContext* ctx)
{
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

std::any ASTBuilder::visitLiteralBoolean(GazpreaParser::LiteralBooleanContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitBool" << ctx->getText() << std::endl;
#endif
    std::shared_ptr<BoolNode> t = std::make_shared<BoolNode>(ctx->getStart()->getLine(), ctx->getText() == "true");

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitLiteralCharacter(GazpreaParser::LiteralCharacterContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitCharacter" << ctx->getText() << std::endl;
#endif
    std::string charContent = ctx->getText().substr(1, ctx->getText().size() - 2); // remove quotes
    try {
        auto charPair = CharNode::consumeChar(charContent);

        auto t = std::make_shared<CharNode>(ctx->getStart()->getLine(), charPair.first);
        return std::dynamic_pointer_cast<ASTNode>(t);
    } catch (std::runtime_error& error) {
        throw SyntaxError(ctx->getStart()->getLine(), error.what());
    }
}

std::any ASTBuilder::visitLiteralString(GazpreaParser::LiteralStringContext* ctx)
{
    // ANTLR escapes backslashes and other chars for us,
    // so I just retrieve the value as is
    // not sure if we'll have to do work to go back and account for escape sequences
#ifdef DEBUG
    std::cout << "visitLiteralString" << ctx->getText() << std::endl;
#endif
    auto t = std::make_shared<VectorNode>(ctx->getStart()->getLine());
    t->isString = true;
    std::string val = ctx->getText().substr(1, ctx->getText().size() - 2); // remove quotes

    // iteratively consume chars until the string is empty
    while (!val.empty()) {
        try {
            auto charPair = CharNode::consumeChar(val);
            auto chNode = std::make_shared<CharNode>(ctx->getStart()->getLine(), charPair.first);
            t->addChild(std::dynamic_pointer_cast<ASTNode>(chNode));
            val = charPair.second;
        } catch (std::runtime_error& error) {
            throw SyntaxError(ctx->getStart()->getLine(), error.what());
        }
    }

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitLiteralVector(GazpreaParser::LiteralVectorContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitLiteralVector" << ctx->getText() << std::endl;
#endif
    return visit(ctx->literal_vector());
}

// this literal_vector is used in both literalVector and literalMatrix to parse vectors
std::any ASTBuilder::visitLiteral_vector(GazpreaParser::Literal_vectorContext* ctx)
{
    auto t = std::make_shared<VectorNode>(ctx->getStart()->getLine());

    for (auto exprCtx : ctx->expression()) {
        t->addChild(visit(exprCtx));
    }

    return std::dynamic_pointer_cast<ASTNode>(t);
}

//    std::any ASTBuilder::visitLiteralMatrix(GazpreaParser::LiteralMatrixContext *ctx) {
// #ifdef DEBUG
//        std::cout << "visitLiteralMatrix" << ctx->getText() << std::endl;
// #endif
//        auto t = std::make_shared<MatrixNode>(ctx->getStart()->getLine());
//
//        auto matrixCtx = ctx->literal_matrix();
//        for (auto vectorCtx : matrixCtx->literal_vector()) {
//            t->addChild(visit(vectorCtx));
//        }
//
//        return std::dynamic_pointer_cast<ASTNode>(t);
//    }

std::any ASTBuilder::visitMath(GazpreaParser::MathContext* ctx)
{
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
    case GazpreaParser::RESERVED_OR:
        t->op = BINOP::OR;
        break;
    default:
        throw std::runtime_error("unknown arithmetic operator " + ctx->op->getText());
    }

    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitCmp(GazpreaParser::CmpContext* ctx)
{
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

std::any ASTBuilder::visitUnary(GazpreaParser::UnaryContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitUnary, op = " << ctx->op->getText() << std::endl;
#endif
    std::shared_ptr<UnaryExpr> t = std::make_shared<UnaryArithNode>(ctx->getStart()->getLine());

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

std::any ASTBuilder::visitLvalue(GazpreaParser::LvalueContext* ctx)
{
    std::shared_ptr<ExprListNode> t = std::make_shared<ExprListNode>(ctx->getStart()->getLine());
    for (auto expr : ctx->expression()) {
        t->addChild(visit(expr));
    }

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitAssign(GazpreaParser::AssignContext* ctx)
{
    std::shared_ptr<AssignNode> t = std::make_shared<AssignNode>(ctx->getStart()->getLine());

    t->addChild(visit(ctx->lvalue()));
    t->addChild(visit(ctx->rvalue));

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitVardecl(GazpreaParser::VardeclContext* ctx)
{
    std::shared_ptr<Symbol> identifierSymbol = std::make_shared<Symbol>(ctx->ID()->getText());
    std::shared_ptr<DeclNode> t = std::make_shared<DeclNode>(ctx->getStart()->getLine(), identifierSymbol);
    if (ctx->qualifier()) {
        t->qualifier = std::any_cast<QUALIFIER>(visit(ctx->qualifier()));
    } else {
        t->qualifier = QUALIFIER::VAR;
    }
    if (ctx->type())
        t->addChild(visit(ctx->type()));
    if (ctx->expression()) {
        t->addChild(visit(ctx->expression()));
    }

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitQualifier(GazpreaParser::QualifierContext* ctx)
{
    if (ctx->RESERVED_CONST()) {
        return QUALIFIER::CONST;
    } else if (ctx->RESERVED_VAR()) {
        return QUALIFIER::VAR;
    } else {
#ifdef DEBUG
        std::cout << "ERROR: unknown qualifier" << ctx->getText() << std::endl;
#endif
        throw std::runtime_error("unknown qualifier " + ctx->getText());
    }
}

std::any ASTBuilder::visitCond(GazpreaParser::CondContext* ctx)
{
#ifdef DEBUGTUPLE
    std::cout << "visitCond " << std::endl;
#endif
    std::shared_ptr<ConditionalNode> t = std::make_shared<ConditionalNode>(ctx->getStart()->getLine());

    for (auto condition : ctx->expression()) {
        t->conditions.push_back(std::any_cast<std::shared_ptr<ASTNode>>(visit(condition)));
    }

    for (auto body : ctx->bodyStatement()) {
        t->bodies.push_back(std::any_cast<std::shared_ptr<ASTNode>>(visit(body)));
    }

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitTupleIndex(GazpreaParser::TupleIndexContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitTupleIndex" << std::endl;
#endif
    std::shared_ptr<ASTNode> t = std::make_shared<TupleIndexNode>(ctx->getStart()->getLine());
    std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID(0)->getSymbol()->getText());
    t->addChild(std::dynamic_pointer_cast<ASTNode>(std::make_shared<IDNode>(ctx->getStart()->getLine(), sym)));
    if (ctx->INT()) {
        t->addChild(std::dynamic_pointer_cast<ASTNode>(std::make_shared<IntNode>(ctx->getStart()->getLine(), std::stoi(ctx->INT()->getText()))));
    } else {
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID(1)->getSymbol()->getText());
        t->addChild(std::dynamic_pointer_cast<ASTNode>(std::make_shared<IDNode>(ctx->getStart()->getLine(), sym)));
    }
    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitIndex(GazpreaParser::IndexContext* ctx)
{
#ifdef DEBUG
    std::cout << "visitIndex" << std::endl;
#endif
    std::shared_ptr<ASTNode> t = std::make_shared<IndexNode>(ctx->getStart()->getLine());

    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));

    if (ctx->expr().size() == 3) {
        // this is matrix index
        t->addChild(visit(ctx->expr(2)));
    }

    return std::dynamic_pointer_cast<ASTNode>(t);
}

std::any ASTBuilder::visitRange(GazpreaParser::RangeContext* ctx)
{
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

std::any ASTBuilder::visitBlock(GazpreaParser::BlockContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting block definition." << std::endl;
#endif
    auto blockNode = std::make_shared<BlockNode>(ctx->getStart()->getLine());

    for (auto statement : ctx->statement()) {
        auto statementNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(statement));
        blockNode->addChild(statementNode);
    }

    return std::dynamic_pointer_cast<ASTNode>(blockNode);
}

std::any ASTBuilder::visitBodyStatement(GazpreaParser::BodyStatementContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting body statement." << std::endl;
#endif
    // a body statement of a conditional or loop is a bit different
    // it can be a block, or any valid one-line statement (not variable declarations)
    // we still want to return a BlockNode though

    // if the body statement is a block, simply return the inner block
    if (ctx->block()) {
        return visit(ctx->block());
    }

    // else, visit the child statement and return a block with that statement
    // there is guaranteed to be only one child statement
    auto blockNode = std::make_shared<BlockNode>(ctx->getStart()->getLine());
    auto innerNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->children[0]));
    blockNode->addChild(innerNode);

    return std::dynamic_pointer_cast<ASTNode>(blockNode);
}

std::any ASTBuilder::visitPredicatedLoop(GazpreaParser::PredicatedLoopContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting predicated loop." << std::endl;
#endif
    std::shared_ptr<ASTNode> t = std::make_shared<PredicatedLoopNode>(ctx->getStart()->getLine());

    t->addChild(visit(ctx->expression()));
    t->addChild(visit(ctx->bodyStatement()));

    return t;
}

std::any ASTBuilder::visitInfiniteLoop(GazpreaParser::InfiniteLoopContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting infinite loop." << std::endl;
#endif
    std::shared_ptr<ASTNode> t = std::make_shared<InfiniteLoopNode>(ctx->getStart()->getLine());

    t->addChild(visit(ctx->bodyStatement()));

    return t;
}

std::any ASTBuilder::visitPostPredicatedLoop(GazpreaParser::PostPredicatedLoopContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting post predicated loop." << std::endl;
#endif
    std::shared_ptr<ASTNode> t = std::make_shared<PostPredicatedLoopNode>(ctx->getStart()->getLine());

    t->addChild(visit(ctx->expression()));
    t->addChild(visit(ctx->bodyStatement()));

    return t;
}

std::any ASTBuilder::visitIteratorLoop(GazpreaParser::IteratorLoopContext* ctx)
{
    std::shared_ptr<IteratorLoopNode> t = std::make_shared<IteratorLoopNode>(ctx->getStart()->getLine());
    // add the 1st domain
    for (int i = 0; i < ctx->ID().size(); i++) {
        auto domain = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression(i)));
        auto domainCasted = std::dynamic_pointer_cast<ExprNode>(domain);
        auto sym = std::make_shared<Symbol>(ctx->ID(i)->getSymbol()->getText());

        t->domainExprs.push_back(std::make_pair(sym, domainCasted));
    }
    t->addChild(visit(ctx->bodyStatement()));
    return std::dynamic_pointer_cast<ASTNode>(t);
}

// Loop Control
std::any ASTBuilder::visitBreak(GazpreaParser::BreakContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting break." << std::endl;
#endif

    // wrap this in a if(true)
    auto ifConditional = std::make_shared<ConditionalNode>(ctx->getStart()->getLine());
    ifConditional->conditions.push_back(std::make_shared<BoolNode>(ctx->getStart()->getLine(), true));

    // create body
    std::shared_ptr<ASTNode> continueNode = std::make_shared<BreakNode>(ctx->getStart()->getLine());
    auto block = std::make_shared<BlockNode>(ctx->getStart()->getLine());
    block->addChild(continueNode);
    ifConditional->bodies.push_back(block);

    return std::dynamic_pointer_cast<ASTNode>(ifConditional);
}

std::any ASTBuilder::visitContinue(GazpreaParser::ContinueContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting continue." << std::endl;
#endif
    // wrap this in a if(true)
    auto ifConditional = std::make_shared<ConditionalNode>(ctx->getStart()->getLine());
    ifConditional->conditions.push_back(std::make_shared<BoolNode>(ctx->getStart()->getLine(), true));

    // create body
    std::shared_ptr<ASTNode> continueNode = std::make_shared<ContinueNode>(ctx->getStart()->getLine());
    auto block = std::make_shared<BlockNode>(ctx->getStart()->getLine());
    block->addChild(continueNode);
    ifConditional->bodies.push_back(block);

    return std::dynamic_pointer_cast<ASTNode>(ifConditional);
}

std::any ASTBuilder::visitProcedure(GazpreaParser::ProcedureContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting procedure definition." << std::endl;
    std::cout << ctx->ID()->getText();
#endif
    auto procSymbol = std::make_shared<Symbol>(ctx->ID()->getText());
    auto procedureNode = std::make_shared<ProcedureNode>(ctx->getStart()->getLine(), procSymbol);

    for (auto arg : ctx->parameter()) {
        auto argResult = std::any_cast<std::shared_ptr<ASTNode>>(visit(arg));

        procedureNode->orderedArgs.push_back(argResult);
    }
    if (ctx->type()) {
        // has return type
        procedureNode->addChild(visit(ctx->type()));
    }

    if (ctx->block()) {
        auto blockResult = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->block()));
        procedureNode->body = blockResult;
    }

    return std::dynamic_pointer_cast<ASTNode>(procedureNode);
}

std::any ASTBuilder::visitFunction(GazpreaParser::FunctionContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting function definition." << std::endl;
#endif
    auto funcSymbol = std::make_shared<Symbol>(ctx->ID()->getText());
    auto functionNode = std::make_shared<FunctionNode>(ctx->getStart()->getLine(), funcSymbol);
    for (auto arg : ctx->funcParameter()) {
        auto argResult = std::any_cast<std::shared_ptr<ASTNode>>(visit(arg));

        functionNode->orderedArgs.push_back(argResult);
    }

    if (ctx->block()) {
        auto blockResult = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->block()));
        functionNode->body = blockResult;
    }
    if (ctx->expression()) {
        auto expr = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));
        auto blockNode = std::make_shared<BlockNode>(ctx->getStart()->getLine());
        auto returnNode = std::make_shared<ReturnNode>(ctx->getStart()->getLine());
        returnNode->returnExpr = expr;

        blockNode->addChild(std::dynamic_pointer_cast<ASTNode>(returnNode));

        functionNode->body = blockNode;
    }
    if (ctx->RESERVED_RETURNS()) {
        functionNode->addChild(visit(ctx->type()));
    }

    return std::dynamic_pointer_cast<ASTNode>(functionNode);
}

std::any ASTBuilder::visitParameter(GazpreaParser::ParameterContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting parameters" << std::endl;
#endif
    auto argNode = std::make_shared<ArgNode>(ctx->getStart()->getLine());

    std::shared_ptr<Symbol> identifierSymbol = std::make_shared<Symbol>(ctx->ID()->getText());

    std::shared_ptr<ASTNode> typeNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->type()));

    argNode->idSym = identifierSymbol;
    argNode->type = typeNode;
    if (ctx->qualifier()) {
        argNode->qualifier = std::any_cast<QUALIFIER>(visit(ctx->qualifier()));
    } else {
        argNode->qualifier = QUALIFIER::CONST;
    }

    return std::dynamic_pointer_cast<ASTNode>(argNode);
}

std::any ASTBuilder::visitFuncParameter(GazpreaParser::FuncParameterContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting function parameter" << std::endl;
#endif

    auto argNode = std::make_shared<ArgNode>(ctx->getStart()->getLine());

    std::shared_ptr<Symbol> identifierSymbol = std::make_shared<Symbol>(ctx->ID()->getText());

    auto typeNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->type()));

    argNode->idSym = identifierSymbol;
    argNode->type = typeNode;
    argNode->qualifier = QUALIFIER::CONST;

    return std::dynamic_pointer_cast<ASTNode>(argNode);
}
std::any ASTBuilder::visitFuncCall(GazpreaParser::FuncCallContext* ctx)
{
    // this is #funcCall rule in expr rule
    return visit(ctx->functionCall());
}

std::any ASTBuilder::visitNormalFunctionCall(GazpreaParser::NormalFunctionCallContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting function call" << std::endl;
#endif
    // functionCall : ID '(' (expression (',' expression)*)? ')'
    //  TODO: how to determine if theh function is normal or not? i guess we do that in other passes when resolving name
    std::shared_ptr<CallNode> callNode = std::make_shared<CallNode>(ctx->getStart()->getLine());
    std::shared_ptr<Symbol> fcallName = std::make_shared<Symbol>(ctx->callable_ids()->getText());

    callNode->CallName = fcallName;
    for (auto expr : ctx->expression()) {
        callNode->addChild(visit(expr));
    }
    return std::dynamic_pointer_cast<ASTNode>(callNode);
}

std::any ASTBuilder::visitNormalProcedureCall(GazpreaParser::NormalProcedureCallContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting procedure call" << std::endl;
#endif
    std::shared_ptr<CallNode> pCallNode = std::make_shared<CallNode>(ctx->getStart()->getLine());
    std::shared_ptr<Symbol> pcallName = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());

    pCallNode->CallName = pcallName;
    for (auto expr : ctx->expression()) {
        pCallNode->addChild(visit(expr));
    }
    pCallNode->procCall = true;
    return std::dynamic_pointer_cast<ASTNode>(pCallNode);
}

std::any ASTBuilder::visitReturn(GazpreaParser::ReturnContext* ctx)
{
#ifdef DEBUG
    std::cout << "Visiting return" << std::endl;
#endif
    auto returnNode = std::make_shared<ReturnNode>(ctx->getStart()->getLine());

    if (ctx->expression()) {
        returnNode->returnExpr = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));
    }

    return std::dynamic_pointer_cast<ASTNode>(returnNode);
}

std::any ASTBuilder::visitGenerator(GazpreaParser::GeneratorContext* ctx)
{
    std::string domainVar1 = ctx->ID(0)->getSymbol()->getText();
    std::string domainVar2 = "";
    if (ctx->ID().size() == 2)
        domainVar2 = ctx->ID(1)->getSymbol()->getText();
    std::shared_ptr<GeneratorNode> gNode = std::make_shared<GeneratorNode>(domainVar1, domainVar2, ctx->getStart()->getLine());
    for (auto expr : ctx->expression()) {
        gNode->addChild(visit(expr));
    }
    return std::dynamic_pointer_cast<ASTNode>(gNode);
}

std::any ASTBuilder::visitFilter(GazpreaParser::FilterContext* ctx)
{
    std::string domainVar = ctx->ID()->getSymbol()->getText();
    std::shared_ptr<FilterNode> fNode = std::make_shared<FilterNode>(domainVar, ctx->getStart()->getLine());
    for (auto expr : ctx->expression()) {
        fNode->addChild(visit(expr));
    }
    return std::dynamic_pointer_cast<ASTNode>(fNode);
}

std::any ASTBuilder::visitConcatenation(GazpreaParser::ConcatenationContext* ctx)
{
    auto concatNode = std::make_shared<ConcatNode>(ctx->getStart()->getLine());
    concatNode->op = CONCAT;

    concatNode->addChild(visit(ctx->expr(0)));
    concatNode->addChild(visit(ctx->expr(1)));
    return std::dynamic_pointer_cast<ASTNode>(concatNode);
}

std::any ASTBuilder::visitStride(GazpreaParser::StrideContext* ctx)
{
    auto strideNode = std::make_shared<StrideNode>(ctx->getStart()->getLine());
    strideNode->op = STRIDE;

    strideNode->addChild(visit(ctx->expr(0)));
    strideNode->addChild(visit(ctx->expr(1)));
    return std::dynamic_pointer_cast<ASTNode>(strideNode);
}
}
