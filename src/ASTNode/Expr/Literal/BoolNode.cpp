#include "ASTNode/Expr/Literal/BoolNode.h"

std::string BoolNode::toString() {
    return "BOOL " + std::to_string(val);
}

bool BoolNode::getVal() {
    return val;
}
