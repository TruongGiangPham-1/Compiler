#include "ASTNode/Expr/Literal/RealNode.h"

std::string RealNode::toString() {
    return "REAL " + std::to_string(val);
}

float RealNode::getVal() {
    return val;
}
