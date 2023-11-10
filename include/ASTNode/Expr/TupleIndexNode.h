//
// Created by truong on 10/11/23.
//


#pragma once
#include "ASTNode/ASTNode.h"
#include "Symbol.h"

// Children - [IDNode, (IDNode | IntNode)]
class TupleIndexNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym; // pointer to symbol definition
    int index = -1;
    TupleIndexNode(int line) : ASTNode(line) {}


    std::shared_ptr<ASTNode> getIDNode() {
        return this->children[0];
    };
    std::shared_ptr<ASTNode> getIndexNode() {
        return this->children[1];
    };
};
