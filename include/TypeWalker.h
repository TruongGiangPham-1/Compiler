#pragma once
#include "ASTNode/ASTNode.h"
#include "ASTWalker.h"
#include <memory>

class TypeWalker : public gazprea::ASTWalker {
  private:
    std::string promotionTable[5][5] = {
/*             boolean   character    integer  real  vector */ 
/*boolean*/  {"boolean", "",     "",    "",   "vector" }, 
/*character*/{"",   "character", "",    "",   "vector" },
/*integer*/  {"",   "",     "integer",  "real",    "vector" },
/*real*/     {"",   "",     "",    "real",    "vector" },
/*vector*/   {"vector",   "vector",   "vector",   "vector",  "vector" }

    };

    const int boolIndex = 0;
    const int charIndex = 1;
    const int integerIndex = 2;
    const int realIndex = 3;
    const int vectorIndex = 4;

    int getTypeIndex(const std::string type);
    bool isListType(const std::shared_ptr<Type> type);
    std::shared_ptr<Type> getPromotedType(const std::shared_ptr<Type> from, const std::shared_ptr<Type> to);
  public:
    std::any visitArith(std::shared_ptr<ArithNode> tree) override;
    std::any visitCmp(std::shared_ptr<CmpNode> tree) override;
    std::any visitIndex(std::shared_ptr<IndexNode> tree) override;
    std::any visitID(std::shared_ptr<IDNode> tree) override;
    std::any visitInt(std::shared_ptr<IntNode> tree) override;
    std::any visitFilter(std::shared_ptr<FilterNode> tree) override;
    std::any visitGenerator(std::shared_ptr<GeneratorNode> tree) override;
    std::any visitRangeVec(std::shared_ptr<RangeVecNode> tree) override;
};
