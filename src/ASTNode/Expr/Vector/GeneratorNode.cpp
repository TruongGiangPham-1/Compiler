#include "ASTNode/Expr/Vector/GeneratorNode.h"

GeneratorNode::GeneratorNode(size_t type, std::string domainVar, int line) : BaseVectorNode(type, line), domainVar(domainVar) {}

std::shared_ptr<ASTNode> GeneratorNode::getVecNode() {
    return children[0];
}

std::shared_ptr<ASTNode> GeneratorNode::getExpr() {
    return children[1];
}
