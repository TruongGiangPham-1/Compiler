#pragma once

#include "GazpreaBaseVisitor.h"

namespace gazprea {

    class ASTBuilder : public GazpreaBaseVisitor {
    public:
        std::any visitFile(GazpreaParser::FileContext *ctx) override;
        std::any visitAssign(GazpreaParser::AssignContext *ctx) override;

        // streams
        std::any visitOutputStream(GazpreaParser::OutputStreamContext *ctx) override;

        // variable declarations
        std::any visitSized(GazpreaParser::SizedContext *ctx) override;
        std::any visitInferred_size(GazpreaParser::Inferred_sizeContext *ctx) override;

        // type stuff
        std::any visitQualifier(GazpreaParser::QualifierContext *ctx) override;
        std::any visitBuilt_in_type(GazpreaParser::Built_in_typeContext *ctx) override;
        // unknown sizes
        std::any visitVector(GazpreaParser::VectorContext *ctx) override;
        std::any visitString(GazpreaParser::StringContext *ctx) override;
        std::any visitMatrixFirst(GazpreaParser::MatrixFirstContext *ctx) override;
        std::any visitMatrixSecond(GazpreaParser::MatrixSecondContext *ctx) override;
        std::any visitMatrix(GazpreaParser::MatrixContext *ctx) override;
        // known sizes
        std::any visitVector_type(GazpreaParser::Vector_typeContext *ctx) override;
        std::any visitString_type(GazpreaParser::String_typeContext *ctx) override;
        std::any visitMatrix_type(GazpreaParser::Matrix_typeContext *ctx) override;
        std::any visitTuple_type(GazpreaParser::Tuple_typeContext *ctx) override;

        // EXPR
        std::any visitExpression(GazpreaParser::ExpressionContext *ctx) override;
        std::any visitIdentity(GazpreaParser::IdentityContext *ctx) override;
        std::any visitNull(GazpreaParser::NullContext *ctx) override;
        std::any visitLiteralID(GazpreaParser::LiteralIDContext *ctx) override;
        std::any visitLiteralInt(GazpreaParser::LiteralIntContext *ctx) override;
        std::any visitMath(GazpreaParser::MathContext *ctx) override;
        std::any visitCmp(GazpreaParser::CmpContext *ctx) override;

        std::any visitCond(GazpreaParser::CondContext *ctx) override;
        std::any visitIndex(GazpreaParser::IndexContext *ctx) override;
        std::any visitRange(GazpreaParser::RangeContext *ctx) override;
        std::any visitGenerator(GazpreaParser::GeneratorContext *ctx) override;
        std::any visitFilter(GazpreaParser::FilterContext *ctx) override;

        std::any visitFunctionSingle(GazpreaParser::FunctionSingleContext *ctx) override;
        std::any visitFunctionBlock(GazpreaParser::FunctionBlockContext *ctx) override;
        std::any visitFunctionForward(GazpreaParser::FunctionForwardContext *ctx) override;

        std::any visitFunction_call(GazpreaParser::Function_callContext *ctx) override;
        std::any visitFuncCall(GazpreaParser::FuncCallContext *ctx) override;

    };

}