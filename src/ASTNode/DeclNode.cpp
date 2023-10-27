#pragma once
#include "ASTNode/DeclNode.h"

std::string DeclNode::toString() {
    return "declare " + getIDName() + " = " + getExprNode()->toString();
}

std::shared_ptr<Symbol> DeclNode::getID() {
    return sym;
}

std::string DeclNode::getIDName() {
    return sym->getName();
}

std::shared_ptr<ASTNode> DeclNode::getTypeNode() {
    return children[0];
}

std::shared_ptr<ASTNode> DeclNode::getExprNode() {
    return children[1];
}
