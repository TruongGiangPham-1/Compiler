#include "ASTNode/Expr/Literal/IDNode.h"

IDNode::IDNode(int line, std::shared_ptr<Symbol> sym) : ExprNode(line), sym(sym) {}

std::string IDNode::toString() {
    return "ID " + getName();
}

std::string IDNode::getName() {
    return sym->getName();
}

std::shared_ptr<Symbol> IDNode::getVal() {
    return sym;
}
