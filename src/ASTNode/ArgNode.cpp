//
// Created by truong on 03/11/23.
//

#include "ASTNode/ArgNode.h"

ArgNode::ArgNode(int line):ASTNode(line) {
}

std::string ArgNode::toString() {
    return "ArgNode";
}
