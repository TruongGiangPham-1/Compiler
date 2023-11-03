#include "ASTNode/Expr/IdentityNode.h"

IdentityNode::IdentityNode(int line) : ExprNode(line) {}

std::string IdentityNode::toString() {
    return "NULL";
}
