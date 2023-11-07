//
// Created by truong on 02/11/23.
//

#ifndef GAZPREABASE_CALLNODE_H
#define GAZPREABASE_CALLNODE_H

#pragma once
#include "ASTNode.h"
#include "FunctionCallTypes/FuncCallType.h"
#include "ScopedSymbol.h"

// Since there isn't much to decouple, I decided to just inherit from ASTNode

class CallNode: public ExprNode {
public:
    //FUNCTYPE functype;
    std::shared_ptr<Symbol> CallName;  // only used for calling user defined function
    std::shared_ptr<ScopedSymbol> MethodRef;  // symbol to the function/procedure definition that it is calling

    CallNode(int loc): ASTNode(loc) {};


    std::string toString() override {
        return "CallNode";
    };
};
#endif //GAZPREABASE_CALLNODE_H
