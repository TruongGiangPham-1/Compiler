#pragma once
#include "ASTNode/Expr/ExprNode.h"

// Children: [expr]
class UnaryExpr : public ExprNode
{
public:
    UNARYOP op;

    UnaryExpr(int line);
    std::shared_ptr<ASTNode> getExpr();

    std::string toString() override;
};
