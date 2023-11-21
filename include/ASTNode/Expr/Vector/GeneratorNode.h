#pragma once
#include "BaseVectorExpr.h"

// [domainVar in vector | expr]
// children: [vector, expr]
class GeneratorNode : public BaseVectorExpr {
public:
    std::string domainVar1;
    std::string domainVar2;
    GeneratorNode(std::string domainVar1, std::string domainVar2, int line);

    std::shared_ptr<Symbol> domainVar1Sym;
    std::shared_ptr<Symbol> domainVar2Sym;
    std::string toString() override;

    std::shared_ptr<ASTNode> getExpr();
    std::shared_ptr<ASTNode> getVectDomain();  // vector generator have 1 domain
    std::pair<std::shared_ptr<ASTNode>, std::shared_ptr<ASTNode>> getMatrixDomain();  // matrix generator have 2 domain
};
