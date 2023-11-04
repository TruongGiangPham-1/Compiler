#pragma once

#include "ASTNode/Expr/ExprNode.h"

// Children: [leftExpr, rightExpr]
// similar to ExprNode, this class should theoretically never be instantiated by itself
class BinaryExpr : public ExprNode
{
public:
    BINOP op;

    BinaryExpr(int line);
    std::shared_ptr<ExprNode> getLHS();
    std::shared_ptr<ExprNode> getRHS();

    // base toString method
    std::string toString() override;
};

// because these next two classes are so short, I'll keep them in this file
// they have no overrides and no new methods

class ArithOpNode : public BinaryExpr {
public:
    ArithOpNode(int line) : BinaryExpr(line){}
};

class CmpOpNode : public BinaryExpr {
public:
    CmpOpNode(int line) : BinaryExpr(line){}
};

class BoolOpNode : public BinaryExpr {
public:
    BoolOpNode(int line) : BinaryExpr(line){}
};

// given a[b], children are [a, b]
class IndexNode : public BinaryExpr {
public:
    IndexNode(int line) : BinaryExpr(line){}
};
