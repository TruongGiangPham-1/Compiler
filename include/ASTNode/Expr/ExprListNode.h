#pragma once
#include "ASTNode/ASTNode.h"

// Class for list of expressions
class ExprListNode : public ASTNode {
public:
    ExprListNode(int line) : ASTNode(line) {}
    std::string toString() override;
};
