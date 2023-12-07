//
// Created by Joshua Ji on 2023-11-20.
//

#include "ASTNode/Expr/Literal/StringNode.h"

StringNode::StringNode(int line) : ExprNode(line), val("") {}

std::string StringNode::toString() {
    return "StringNode: " + val;
}

std::string StringNode::getVal() {
    return val;
}


int StringNode::getSize() {
    return size;
}