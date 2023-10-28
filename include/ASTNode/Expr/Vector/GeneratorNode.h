#pragma once
#include "BaseVectorExpr.h"

// [domainVar in vector | expr]
// children: [vector, expr]
class GeneratorNode : public BaseVectorExpr {
public:
    std::string domainVar;
    GeneratorNode(std::string domainVar, int line);

    std::string toString() override;

    std::shared_ptr<ASTNode> getVecNode();
    std::shared_ptr<ASTNode> getExpr();
};
