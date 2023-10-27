#include "ASTNode/PrintNode.h"

PrintNode::PrintNode(int line) : ASTNode(line) {}

std::shared_ptr<ASTNode> PrintNode::getExpr() {
    return children[0];
}
