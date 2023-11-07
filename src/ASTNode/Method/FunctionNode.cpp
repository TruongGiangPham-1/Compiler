//
// Created by truong on 01/11/23.
//

#include "ASTNode/Method/FunctionNode.h"

FunctionNode::FunctionNode(int line, std::shared_ptr<Symbol>funcNameSym): ASTNode(line), funcNameSym(funcNameSym) {}


std::string FunctionNode::toString() {
    if (body) {
        return "Function " + funcNameSym->getName() + " " + body->toStringTree();
    } else if (expr){
        return "Function " + funcNameSym->getName() + " " + expr->toStringTree();
    } else {
        return "Function " + funcNameSym->getName();
    }
}

std::shared_ptr<ASTNode> FunctionNode::getRetTypeNode() {
    return this->children[0];
}
