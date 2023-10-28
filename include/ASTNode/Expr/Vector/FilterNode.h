#pragma once
#include "BaseVectorExpr.h"

// [domainVar in vector & expr]
// children: [vector, expr]
class FilterNode : public BaseVectorExpr {
public:
    std::string domainVar;
    FilterNode(std::string domainVar, int line);

    std::string toString() override;

    std::shared_ptr<ASTNode> getVecNode();
    std::shared_ptr<ASTNode> getExpr();
};
