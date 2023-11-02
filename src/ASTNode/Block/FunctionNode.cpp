//
// Created by truong on 01/11/23.
//

#include "../include/ASTNode/Block/FunctionNode.h"



FunctionNode::FunctionNode(int line, std::shared_ptr<Symbol>funcNameSym): BlockNode(line), funcNameSym(funcNameSym) {}


std::string FunctionNode::toString() {
    return "Function";
}

std::shared_ptr<ASTNode> FunctionNode::getRetTypeNode() {
    return this->children[0];
}

// === FunctionBlock
std::shared_ptr<ASTNode> FunctionBlockNode::getBlock() {
    return this->children[1];
}

// === Function Single node
std::shared_ptr<ASTNode>FunctionSingleNode::getExpr() {
    return this->children[1];
}