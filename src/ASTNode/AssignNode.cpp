#include "ASTNode/AssignNode.h"

AssignNode::AssignNode(int line, std::shared_ptr<Symbol> sym) : ASTNode(line), sym(sym) {};

std::string AssignNode::toString() {
    return "assign " + getLvalue()->toString() + " = " + getRvalue()->toString();
}

std::string AssignNode::getIDName() {
    return sym->getName();
}

std::shared_ptr<Symbol> AssignNode::getID() {
    return sym;
}

std::shared_ptr<ASTNode> AssignNode::getLvalue() {
    assert(children.size() == 1);
    return this->children[0];
}
std::shared_ptr<ASTNode> AssignNode::getRvalue() {
    assert(children.size() == 2);
    return this->children[1];
}
