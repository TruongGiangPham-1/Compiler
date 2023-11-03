#include "ASTNode/DeclNode.h"

DeclNode::DeclNode(int line, std::shared_ptr<Symbol> sym) : ASTNode(line), sym(sym) {}

std::string DeclNode::toString() {
    return "DECLARE";
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

QUALIFIER DeclNode::getQualifier() {
    return qualifier;
}
