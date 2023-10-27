#include "ASTNode/PrintNode.h"

PrintNode::PrintNode(size_t type, int line) : ASTNode(type, line) {}

std::shared_ptr<ASTNode> PrintNode::getExpr() {
    return children[0];
}
