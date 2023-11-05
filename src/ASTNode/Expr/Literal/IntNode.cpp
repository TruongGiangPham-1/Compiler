#include "ASTNode/Expr/Literal/IntNode.h"

IntNode::IntNode(int line, int val) : ExprNode(line), val(val) {}

std::string IntNode::toString() {
    return "INT " + std::to_string(val);
}

int IntNode::getVal() {
    return val;
}
