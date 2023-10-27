#pragma once

#include "ASTNode/AssignNode.h"

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
    return children[0];
}
