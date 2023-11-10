//
// Created by truong on 10/11/23.
//


#pragma once
#include "ASTNode/ASTNode.h"

// Children - [IDNode, (IDNode | IntNode)]
class TupleIndexNode : public ASTNode {
public:
    TupleIndexNode(int line) : ASTNode(line) {}
};
