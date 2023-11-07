#include "ASTNode/Expr/NullNode.h"

NullNode::NullNode(int line) : ExprNode(line) {}

std::string NullNode::toString() {
    return "NULL";
}
