//
// Created by truong on 01/11/23.
//

#include "../include/ASTNode/Block/FunctionNode.h"



FunctionNode::FunctionNode(int line): BlockNode(line) {}


std::string FunctionNode::toString() {
    return "Function";
}
