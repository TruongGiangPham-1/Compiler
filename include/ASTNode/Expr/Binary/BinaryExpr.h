#pragma once

#include "ASTNode/Expr/ExprNode.h"

// Children: [leftExpr, rightExpr]
// similar to ExprNode, this class should theoretically never be instantiated by itself
class BinaryExpr : public ExprNode
{
public:
    BINOP op;

    BinaryExpr(size_t tokenType, int line) : ExprNode(tokenType, line) {}
    std::shared_ptr<ASTNode> getLHS();
    std::shared_ptr<ASTNode> getRHS();

    // base toString method
    std::string toString() override;
};

// because these next two classes are so short, I'll keep them in this file
// they have no overrides and no new methods

class ArithNode : public BinaryExpr {
public:
    ArithNode(size_t tokenType, int line) : BinaryExpr(tokenType, line){}
};

class CmpNode : public BinaryExpr {
public:
    CmpNode(size_t tokenType, int line) : BinaryExpr(tokenType, line){}
};

// given a[b], children are [a, b]
class IndexNode : public BinaryExpr {
public:
    IndexNode(size_t tokenType, int line) : BinaryExpr(tokenType, line){}
};