#pragma once
#include "ASTNode/Expr/ExprNode.h"

// `val` is calculated in the first pass (Builder)
class CharNode : public ExprNode {
public:
    char val;

    CharNode(int line, int val) : ExprNode(line), val(val) {};

    std::string toString() override;
    char getVal();
};
