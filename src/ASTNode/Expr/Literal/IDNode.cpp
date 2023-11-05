#include "ASTNode/Expr/Literal/IDNode.h"

std::string IDNode::toString() {
    return "ID " + getName();
}

std::string IDNode::getName() {
    return sym->getName();
}

std::shared_ptr<Symbol> IDNode::getVal() {
    return sym;
}
