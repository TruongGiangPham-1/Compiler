#include "ASTNode/AssignNode.h"

AssignNode::AssignNode(int line, std::shared_ptr<Symbol> sym) : ASTNode(line), sym(sym) {};

std::string AssignNode::toString() {
    return "assign " + getIDName() + " = " + getExprNode()->toString();
}

std::string AssignNode::getIDName() {
    return sym->getName();
}

std::shared_ptr<Symbol> AssignNode::getID() {
    return sym;
}

std::shared_ptr<ASTNode> AssignNode::getExprNode() {
    return children[1];
}

std::shared_ptr<ASTNode> AssignNode::getLValue() {
    return children[0];
}


