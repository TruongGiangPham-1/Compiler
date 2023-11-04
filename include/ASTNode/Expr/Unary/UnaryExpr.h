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

class UnaryArithNode : public UnaryExpr {
public:
    UnaryArithNode(int line) : UnaryExpr(line){}
};

class UnaryBoolNode : public UnaryExpr {
public:
    UnaryBoolNode(int line) : UnaryExpr(line){}
};