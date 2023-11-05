#pragma once
#include "ASTNode/Expr/ExprNode.h"

// `val` is calculated in the first pass (Builder)
class RealNode : public ExprNode {
public:
    float val;

    RealNode(int line, float val) : ExprNode(line), val(val) {};

    std::string toString() override;
    float getVal();
};
