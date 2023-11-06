#pragma once
#include "ASTNode/ASTNode.h"
#include "ASTWalker.h"
#include "SymbolTable.h"
#include "Type.h"
#include <memory>

//class TypeWalker : public gazprea::ASTWalker {
//  private:
//    std::string promotionTable[5][5] = {
///*             boolean   character    integer  real  vector */
///*boolean*/  {"boolean",  "",         "",         "",       "vector" },
///*character*/{"",         "character","",         "",       "vector" },
///*integer*/  {"",         "",         "integer",  "real",   "vector" },
///*real*/     {"",         "",         "",         "real",   "vector" },
///*vector*/   {"vector",   "vector",   "vector",   "vector", "vector" }
//
//    };
//
//    const int boolIndex = 0;
//    const int charIndex = 1;
//    const int integerIndex = 2;
//    const int realIndex = 3;
//    const int vectorIndex = 4;
//
//    int getTypeIndex(const std::string type);
//    bool isListType(const std::shared_ptr<Type> type);
//    std::shared_ptr<Type> getPromotedType(const std::shared_ptr<Type> from, const std::shared_ptr<Type> to);
//  public:
//    std::any visitID(std::shared_ptr<IDNode> tree) override;
//
//    std::any visitArith(std::shared_ptr<BinaryArithNode> tree) override;
//    std::any visitCmp(std::shared_ptr<BinaryCmpNode> tree) override;
//    std::any visitIndex(std::shared_ptr<IndexNode> tree) override;
//
//    std::any visitInt(std::shared_ptr<IntNode> tree) override;
//    std::any visitFilter(std::shared_ptr<FilterNode> tree) override;
//    std::any visitGenerator(std::shared_ptr<GeneratorNode> tree) override;
//    std::any visitRangeVec(std::shared_ptr<RangeVecNode> tree) override;
//};

namespace gazprea {
    class PromotedType {
    public:
        std::shared_ptr<SymbolTable> symtab;
        std::shared_ptr<Scope> currentScope;
        static std::string booleanResult[5][5]; //XOR AND NOT OR
        static std::string arithmeticResult[5][5]; //+ - / * ^ % **
        static std::string comparisonResult[5][5]; //>= <= > <
        static std::string equalityResult[5][5]; //==, !=
        static std::string promotionTable[5][5];

        const int boolIndex = 0;
        const int charIndex = 1;
        const int integerIndex = 2;
        const int realIndex = 3;
        const int tupleIndex = 4;

        std::shared_ptr<Type> getType(std::string table[5][5], std::shared_ptr<ASTNode> lhs, std::shared_ptr<ASTNode> rhs, std::shared_ptr<ASTNode> t);
        int getTypeIndex(const std::string type);

        PromotedType(std::shared_ptr<SymbolTable> symtab);
        ~PromotedType();
    };


    class TypeWalker : public ASTWalker {
    private:
        std::shared_ptr<SymbolTable> symtab;
        std::shared_ptr<Scope> currentScope;
        std::shared_ptr<PromotedType> promotedType;

    public:
        TypeWalker(std::shared_ptr<SymbolTable> symtab, std::shared_ptr<PromotedType> promotedType);
        ~TypeWalker();

        std::any visitAssign(std::shared_ptr<AssignNode> tree) override;
        std::any visitDecl(std::shared_ptr<DeclNode> tree) override;

        //std::any visitID(std::shared_ptr<IDNode> tree) override;
        std::any visitInt(std::shared_ptr<IntNode> tree) override;
        std::any visitReal(std::shared_ptr<RealNode> tree) override;
        //std::any visitTuple(std::shared_ptr<TupleNode> tree) override;
        std::any visitChar(std::shared_ptr<CharNode> tree) override;
        std::any visitBool(std::shared_ptr<BoolNode> tree) override;

        std::any visitArith(std::shared_ptr<BinaryArithNode> tree) override;
        std::any visitCmp(std::shared_ptr<BinaryCmpNode> tree) override;
        std::any visitUnaryArith(std::shared_ptr<UnaryArithNode>tree) override;
    };



}
