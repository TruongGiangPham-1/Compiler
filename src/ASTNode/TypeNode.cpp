#include "ASTNode/TypeNode.h"

std::string TypeNode::getTypeName() {
    return sym->getName();
}

std::string TypeNode::toString() {
    return "Type: " + getTypeName();
}