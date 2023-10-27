#include "ASTNode/Expr/IDNode.h"

std::string IDNode::toString() {
    return "ID " + getName();
}

std::string IDNode::getName() {
    return sym->getName();
}