//
// Created by truong on 01/11/23.
//

#include "ASTNode/Method/FunctionNode.h"

FunctionNode::FunctionNode(int line, std::shared_ptr<Symbol>funcNameSym): ASTNode(line), funcNameSym(funcNameSym) {}


std::string FunctionNode::toString() {
    return "Function " + funcNameSym->getName() + " " + body->toStringTree();
}

std::shared_ptr<ASTNode> FunctionNode::getRetTypeNode() {
    return this->children[0];
}
