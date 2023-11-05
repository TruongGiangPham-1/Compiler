#pragma once
#include "ASTNode/Expr/ExprNode.h"

// `val` is calculated in the first pass (Builder)
class StringNode: public ExprNode {
public:
    std::string val;

    StringNode(int line, std::string val) : ExprNode(line), val(val) {};

    std::string toString() override;
    std::string getVal();
};
