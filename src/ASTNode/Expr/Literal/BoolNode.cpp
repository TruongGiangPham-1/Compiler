#include "ASTNode/Expr/Literal/BoolNode.h"

BoolNode::BoolNode(int line, bool val) : ExprNode(line), val(val) {}

std::string BoolNode::toString() {
    return "BOOL " + std::to_string(val);
}

bool BoolNode::getVal() {
    return val;
}
