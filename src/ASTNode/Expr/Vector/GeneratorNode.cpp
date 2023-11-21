#include "ASTNode/Expr/Vector/GeneratorNode.h"

GeneratorNode::GeneratorNode(std::string domainVar1, std::string domainVar2 ,int line) : BaseVectorExpr(line), domainVar1(domainVar1), domainVar2(domainVar2) {}

std::string GeneratorNode::toString() {
    return "Generator " + domainVar1 + " " + domainVar2;
}



std::shared_ptr<ASTNode> GeneratorNode::getExpr() {
    assert(!children.empty());
    return children[children.size() - 1];  // return the last expr node
}

std::shared_ptr<ASTNode> GeneratorNode::getVectDomain() {
    if (children.size() == 2) {
        return children[0];
    } else {
        return nullptr;
    }
}

std::pair<std::shared_ptr<ASTNode>, std::shared_ptr<ASTNode>>  GeneratorNode::getMatrixDomain() {
    if (children.size() == 3) {
        return std::make_pair(children[0], children[1]);
    } else {
        return std::make_pair(nullptr, nullptr);
    }
}