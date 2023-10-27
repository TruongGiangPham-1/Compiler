#include "ASTNode/Expr/Vector/GeneratorNode.h"

GeneratorNode::GeneratorNode(std::string domainVar, int line) : BaseVectorNode(line), domainVar(domainVar) {}

std::shared_ptr<ASTNode> GeneratorNode::getVecNode() {
    return children[0];
}

std::shared_ptr<ASTNode> GeneratorNode::getExpr() {
    return children[1];
}
