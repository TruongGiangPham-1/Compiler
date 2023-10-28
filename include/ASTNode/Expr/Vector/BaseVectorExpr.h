#pragma once
#include "ASTNode/Expr/ExprNode.h"

// should never be instantiated by itself
class BaseVectorExpr : public ExprNode {
public:
    BaseVectorExpr(int line) : ExprNode(line) {}
};
