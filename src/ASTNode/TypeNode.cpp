#include "ASTNode/TypeNode.h"

TypeNode::TypeNode(int line, std::shared_ptr<Symbol> sym) : ASTNode(line), sym(sym) {};

std::string TypeNode::getTypeName() {
    return sym->getName();
}

std::string TypeNode::toString() {
    return "Type: " + getTypeName();
}