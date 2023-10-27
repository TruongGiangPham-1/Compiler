#pragma once
#include "BaseVectorNode.h"

// [domainVar in vector | expr]
// children: [vector, expr]
class GeneratorNode : public BaseVectorNode {
public:
    std::string domainVar;
    GeneratorNode(std::string domainVar, int line);

    std::shared_ptr<ASTNode> getVecNode();
    std::shared_ptr<ASTNode> getExpr();
};
