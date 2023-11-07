//
// Created by Joshua Ji on 2023-11-06.
//

#include "ASTNode/ContinueNode.h"

ContinueNode::ContinueNode(int line) : ASTNode(line) {}

std::string ContinueNode::toString() {
    return "Continue";
}
