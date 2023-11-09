#include "ASTNode/AssignNode.h"

AssignNode::AssignNode(int line) : ASTNode(line) {};

std::string AssignNode::toString() {
    return "assign " + getLvalue()->toString() + " = " + getRvalue()->toString();
}

std::shared_ptr<ASTNode> AssignNode::getLvalue() {
    assert(children.size() > 0);
    return this->children[0];
}
std::shared_ptr<ASTNode> AssignNode::getRvalue() {
    assert(children.size() > 1);
    return this->children[1];
}


