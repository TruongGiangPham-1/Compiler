//
// Created by truong on 10/11/23.
//


#pragma once
#include "ASTNode/ASTNode.h"
#include "ASTNode/Expr/ExprNode.h"
#include "Symbol.h"

// Children - [IDNode, (IDNode | IntNode)]
class TupleIndexNode : public ExprNode {
public:
    std::shared_ptr<Symbol> sym; // pointer to symbol definition
    int index = -1;
    TupleIndexNode(int line) : ExprNode(line) {}


    std::shared_ptr<ASTNode> getIDNode() {
        return this->children[0];
    };
    std::shared_ptr<ASTNode> getIndexNode() {
        return this->children[1];
    };
};
