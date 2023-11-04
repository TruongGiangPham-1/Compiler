//
// Created by truong on 03/11/23.
//

#include "ASTNode/ArgNode.h"

ArgNode::ArgNode(int line):ASTNode(line) {
}

std::shared_ptr<ASTNode> ArgNode::getArgType() {
    assert(this->children.size() > 0);
    return this->children[0];
}


std::string ArgNode::toString() {
    return "ArgNode";
}