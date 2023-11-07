//
// Created by truong on 02/11/23.
//

#ifndef GAZPREABASE_FUNCTIONCALLNODE_H
#define GAZPREABASE_FUNCTIONCALLNODE_H

#pragma once
#include "ASTNode.h"
#include "FunctionCallTypes/FuncCallType.h"
#include "ScopedSymbol.h"

// Since there isn't much to decouple, I decided to just inherit from ASTNode

class FunctionCallNode: public ASTNode {
public:
    FUNCTYPE functype;
    std::shared_ptr<Symbol> funcCallName;  // only used for calling user defined function
    std::shared_ptr<FunctionSymbol> functionRef;  // symbol to the function definition that it is calling

    FunctionCallNode(int loc, FUNCTYPE functype): ASTNode(loc), functype(functype) {};


    std::string toString() override {
        return funcTypeStr[functype];
    };
};
#endif //GAZPREABASE_FUNCTIONCALLNODE_H
