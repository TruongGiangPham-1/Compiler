#pragma once
#include "ASTNode/Expr/ExprNode.h"
#include <cwchar>

// `val` is calculated in the first pass (Builder)
class IntNode : public ExprNode {
public:
    int val;

    IntNode(int line, int val) : ExprNode(line), val(val) {};

    std::string toString() override;
    int getVal();
};
