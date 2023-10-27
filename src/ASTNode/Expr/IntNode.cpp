#include "ASTNode/Expr/IntNode.h"

std::string IntNode::toString() {
    return "INT " + std::to_string(val);
}

int IntNode::getVal() {
    return val;
}
