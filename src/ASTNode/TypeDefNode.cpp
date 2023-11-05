#include "ASTNode/TypeDefNode.h"

TypeDefNode::TypeDefNode(int line, std::shared_ptr<Symbol> sym) : ASTNode(line), sym(sym) {};

std::string TypeDefNode::getName() {
    return sym->getName();
}

std::shared_ptr<TypeNode> TypeDefNode::getType() {
    return std::dynamic_pointer_cast<TypeNode>(children[0]);
}

std::string TypeDefNode::toString() {
    return "TypeDefNode " + sym->getName();
}
