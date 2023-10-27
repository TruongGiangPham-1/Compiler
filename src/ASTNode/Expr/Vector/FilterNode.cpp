#include "ASTNode/Expr/Vector/FilterNode.h"

FilterNode::FilterNode(size_t type, std::string domainVar, int line) : BaseVectorNode(type, line), domainVar(domainVar) {}

std::shared_ptr<ASTNode> FilterNode::getVecNode() {
    return children[0];
}

std::shared_ptr<ASTNode> FilterNode::getExpr() {
    return children[1];
}