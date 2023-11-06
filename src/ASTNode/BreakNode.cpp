//
// Created by Joshua Ji on 2023-11-06.
//

#include "ASTNode/BreakNode.h"

BreakNode::BreakNode(int line) : ASTNode(line) {}

std::string BreakNode::toString() {
    return "Break";
}