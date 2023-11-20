//
// Created by Joshua Ji on 2023-11-20.
//

#include "ASTNode/Expr/Literal/StringNode.h"

StringNode::StringNode(int line) : ExprNode(line), val("uninitialized!!") {}

std::string StringNode::toString() {
    return "StringNode: " + val;
}

std::string StringNode::getVal() {
    return val;
}
