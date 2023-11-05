#include "ASTNode/Expr/Literal/RealNode.h"

RealNode::RealNode(int line, float val) : ExprNode(line), val(val) {}

std::string RealNode::toString() {
    return "REAL " + std::to_string(val);
}

float RealNode::getVal() {
    return val;
}
