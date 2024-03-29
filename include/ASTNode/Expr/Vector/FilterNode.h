#pragma once
#include "BaseVectorExpr.h"

// [domainVar in vector & expr]
// children: [vector, expr]
class FilterNode : public BaseVectorExpr {
public:
    std::string domainVar;
    std::shared_ptr<Symbol>domainVarSym;
    FilterNode(std::string domainVar, int line);

    std::string toString() override;

    std::shared_ptr<ASTNode> getDomain();
    std::vector<std::shared_ptr<ASTNode>> getExprList();
};
