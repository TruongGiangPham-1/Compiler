#pragma once

#include "GazpreaBaseVisitor.h"

namespace gazprea {

    class ASTBuilder : public GazpreaBaseVisitor {
    public:
        std::any visitFile(GazpreaParser::FileContext *ctx) override;
        //std::any visitVardecl(GazpreaParser::VardeclContext *ctx) override;
        std::any visitAssign(GazpreaParser::AssignContext *ctx) override;
        std::any visitCond(GazpreaParser::CondContext *ctx) override;
        //std::any visitLoop(GazpreaParser::LoopContext *ctx) override;
        //std::any visitPrint(GazpreaParser::PrintContext *ctx) override;
        //std::any visitParen(GazpreaParser::ParenContext *ctx) override;
        std::any visitIndex(GazpreaParser::IndexContext *ctx) override;
        std::any visitRange(GazpreaParser::RangeContext *ctx) override;
        std::any visitGenerator(GazpreaParser::GeneratorContext *ctx) override;
        std::any visitFilter(GazpreaParser::FilterContext *ctx) override;
        std::any visitMath(GazpreaParser::MathContext *ctx) override;
        std::any visitCmp(GazpreaParser::CmpContext *ctx) override;
        std::any visitLiteralID(GazpreaParser::LiteralIDContext *ctx) override;
        std::any visitLiteralInt(GazpreaParser::LiteralIntContext *ctx) override;
        std::any visitType(GazpreaParser::TypeContext *ctx) override;
        std::any visitExpression(GazpreaParser::ExpressionContext *ctx) override;

        std::any visitFunctionSingle(GazpreaParser::FunctionSingleContext *ctx) override;
        std::any visitFunctionBlock(GazpreaParser::FunctionBlockContext *ctx) override;
        std::any visitFunctionForward(GazpreaParser::FunctionForwardContext *ctx) override;

    };

}