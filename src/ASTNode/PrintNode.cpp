#include "ASTNode/PrintNode.h"
#include <memory>

PrintNode::PrintNode(int line) : ASTNode(line) {}

std::shared_ptr<ExprNode> PrintNode::getExpr() {
    return std::dynamic_pointer_cast<ExprNode>(children[0]);
}

std::string PrintNode::toString() {
    return "Print";
}
