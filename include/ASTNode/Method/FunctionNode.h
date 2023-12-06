//
// Created by truong on 01/11/23.
//

#ifndef GAZPREABASE_FUNCTIONNODE_H
#define GAZPREABASE_FUNCTIONNODE_H

#include "ASTNode/ASTNode.h"
#include "ASTNode/ArgNode.h"
#include "ScopedSymbol.h"
#pragma once

#include "ASTNode/Block/BlockNode.h"
#include "Symbol.h"
#include "Type.h"
class FunctionNode : public ASTNode {
public:
    std::vector<std::shared_ptr<ASTNode>>orderedArgs;    // array of arguments's ID node
    std::vector<std::pair<std::string, int>>declaredVars;  // all the variable declared in this mehtod
    std::shared_ptr<ASTNode> body;
    std::shared_ptr<ASTNode> expr;
    std::shared_ptr<Symbol> funcNameSym;
    FunctionNode(int line, std::shared_ptr<Symbol>funcNameSym);
    std::string toString() override;
    std::shared_ptr<ASTNode> getRetTypeNode();
};

#endif //GAZPREABASE_FUNCTIONNODE_H
