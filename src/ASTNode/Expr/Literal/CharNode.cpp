#include "ASTNode/Expr/Literal/CharNode.h"

std::string CharNode::toString() {
    return "Char Node";
}

char CharNode::getVal() {
    return val;
}
