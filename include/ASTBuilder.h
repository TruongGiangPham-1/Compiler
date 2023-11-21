#pragma once

#include <memory>
#include "GazpreaBaseVisitor.h"

#include "ASTNode/ASTNode.h"
#include "ASTNode/AssignNode.h"
#include "ASTNode/Block/BlockNode.h"
#include "ASTNode/DeclNode.h"
#include "ASTNode/TypeDefNode.h"
#include "ASTNode/Stream/StreamOut.h"
#include "ASTNode/Stream/StreamIn.h"

#include "ASTNode/Expr/NullNode.h"
#include "ASTNode/Expr/Binary/BinaryExpr.h"
#include "ASTNode/Expr/Vector/RangeVecNode.h"
#include "ASTNode/Expr/Vector/GeneratorNode.h"
#include "ASTNode/Expr/Vector/FilterNode.h"
#include "ASTNode/Expr/Unary/UnaryExpr.h"
#include "ASTNode/Expr/ExprListNode.h"

#include "ASTNode/Loop/LoopNode.h"
#include "ASTNode/Loop/PredicatedLoopNode.h"
#include "ASTNode/Loop/IteratorLoopNode.h"
#include "ASTNode/Loop/PostPredicatedLoopNode.h"
#include "ASTNode/Loop/InfiniteLoopNode.h"
#include "ASTNode/BreakNode.h"
#include "ASTNode/ContinueNode.h"

#include "ASTNode/Block/ConditionalNode.h"
#include "ASTNode/Method/FunctionNode.h"
#include "ASTNode/Method/ReturnNode.h"
#include "ASTNode/Type/TypeNode.h"
#include "ASTNode/Type/VectorTypeNode.h"
#include "ASTNode/Type/StringTypeNode.h"
#include "ASTNode/Type/MatrixTypeNode.h"
#include "FunctionCallTypes/FuncCallType.h"
#include "ASTNode/CallNode.h"
#include "ASTNode/Type/TupleTypeNode.h"
#include "ASTNode/Method/ProcedureNode.h"
#include "Types/QUALIFIER.h"
#include "ASTNode/Expr/Literal/IDNode.h"
#include "ASTNode/Expr/Literal/IntNode.h"
#include "ASTNode/Expr/Literal/RealNode.h"
#include "ASTNode/Expr/Literal/TupleNode.h"
#include "ASTNode/Expr/Literal/StringNode.h"
#include "ASTNode/Expr/Literal/BoolNode.h"
#include "ASTNode/Expr/Literal/CharNode.h"
#include "ASTNode/Expr/Literal/VectorNode.h"
#include "ASTNode/Expr/Literal/MatrixNode.h"
#include "ASTNode/Expr/IdentityNode.h"

namespace gazprea {

    class ASTBuilder : public GazpreaBaseVisitor {
    public:
        std::any visitFile(GazpreaParser::FileContext *ctx) override;

        // ()
        std::any visitParentheses(GazpreaParser::ParenthesesContext *ctx) override;
        // memory
        std::any visitVardecl(GazpreaParser::VardeclContext *ctx) override;
        std::any visitQualifier(GazpreaParser::QualifierContext *ctx) override;
        std::any visitAssign(GazpreaParser::AssignContext *ctx) override;
        std::any visitLvalue(GazpreaParser::LvalueContext *ctx) override;
        std::any visitTupleIndex(GazpreaParser::TupleIndexContext *ctx) override;

        // typing
        std::any visitBaseType(GazpreaParser::BaseTypeContext *ctx) override;
        std::any visitTypedef(GazpreaParser::TypedefContext *ctx) override;
        std::any visitVectorType(GazpreaParser::VectorTypeContext *ctx) override;
        std::any visitMatrixType(GazpreaParser::MatrixTypeContext *ctx) override;
        std::any visitTupleType(GazpreaParser::TupleTypeContext *ctx) override;

        std::any visitBlock(GazpreaParser::BlockContext *ctx) override;
        std::any visitBodyStatement(GazpreaParser::BodyStatementContext *ctx) override;


        // literals. will populate ast with values
        std::any visitLiteralInt(GazpreaParser::LiteralIntContext *ctx) override;
        std::any visitLiteralCharacter(GazpreaParser::LiteralCharacterContext *ctx) override;
        std::any visitLiteralReal(GazpreaParser::LiteralRealContext *ctx) override;
        std::any visitLiteralBoolean(GazpreaParser::LiteralBooleanContext *ctx) override;
        std::any visitLiteralTuple(GazpreaParser::LiteralTupleContext *ctx) override;
        std::any visitLiteralID(GazpreaParser::LiteralIDContext *ctx) override;
        std::any visitLiteralString(GazpreaParser::LiteralStringContext *ctx) override;
        std::any visitLiteralVector(GazpreaParser::LiteralVectorContext *ctx) override;
        std::any visitLiteralMatrix(GazpreaParser::LiteralMatrixContext *ctx) override;
        std::any visitLiteral_vector(GazpreaParser::Literal_vectorContext *ctx) override;

        // operations
        std::any visitMath(GazpreaParser::MathContext *ctx) override;
        std::any visitCmp(GazpreaParser::CmpContext *ctx) override;
        std::any visitUnary(GazpreaParser::UnaryContext *ctx) override;
        std::any visitIdentity(GazpreaParser::IdentityContext *ctx) override;
        std::any visitNull(GazpreaParser::NullContext *ctx) override;
        std::any visitIndex(GazpreaParser::IndexContext *ctx) override;
        std::any visitCast(GazpreaParser::CastContext *ctx) override;

        // functions
        std::any visitProcedure(GazpreaParser::ProcedureContext *ctx) override;
        std::any visitFunction(GazpreaParser::FunctionContext *ctx) override;
        std::any visitParameter(GazpreaParser::ParameterContext *ctx) override;
        std::any visitFuncParameter(GazpreaParser::FuncParameterContext *ctx) override;
        std::any visitReturn(GazpreaParser::ReturnContext *ctx) override;

        std::any visitProcedureCall(GazpreaParser::ProcedureCallContext *ctx) override;
        std::any visitFunctionCall(GazpreaParser::FunctionCallContext *ctx) override;  // this is functioncall rule
        std::any visitFuncCall(GazpreaParser::FuncCallContext *ctx) override;   // this is #funcCall label in expr

        // control flow
        std::any visitCond(GazpreaParser::CondContext *ctx) override;

        std::any visitPredicatedLoop(GazpreaParser::PredicatedLoopContext *ctx) override;
        std::any visitInfiniteLoop(GazpreaParser::InfiniteLoopContext *ctx) override;
        std::any visitPostPredicatedLoop(GazpreaParser::PostPredicatedLoopContext *ctx) override;
        std::any visitIteratorLoop(GazpreaParser::IteratorLoopContext *ctx) override;

        // Loop Control
        std::any visitBreak(GazpreaParser::BreakContext *ctx) override;
        std::any visitContinue(GazpreaParser::ContinueContext *ctx) override;

        std::any visitStreamIn(GazpreaParser::StreamInContext *ctx) override;
        std::any visitStreamOut(GazpreaParser::StreamOutContext *ctx) override;

        std::any visitRange(GazpreaParser::RangeContext *ctx) override;
        std::any visitGenerator(GazpreaParser::GeneratorContext *ctx) override;
        std::any visitFilter(GazpreaParser::FilterContext *ctx) override;
    };

}
