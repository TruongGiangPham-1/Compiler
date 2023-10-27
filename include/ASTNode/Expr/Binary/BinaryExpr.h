#pragma once

#include "ASTNode/Expr/ExprNode.h"

// Children: [leftExpr, rightExpr]
class BinaryExpr : public ExprNode
{
public:
    BINOP op;

    BinaryExpr(size_t tokenType) : ExprNode(tokenType) {}
    std::shared_ptr<ASTNode> getLHS();
    std::shared_ptr<ASTNode> getRHS();

    // base toString method
    std::string toString() override;
};

// because these next two classes are so short, I'll keep them in this file
// they have no overrides and no new methods

class BinaryArithNode : public BinaryExpr {
public:
    BinaryArithNode(size_t tokenType) : BinaryExpr(tokenType){}
};

class BinaryCmpNode : public BinaryExpr {
public:
    BinaryCmpNode(size_t tokenType) : BinaryExpr(tokenType){}
};
