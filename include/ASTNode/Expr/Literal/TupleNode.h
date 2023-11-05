#pragma once
#include "ASTNode/Expr/ExprNode.h"

// `val` is calculated in the first pass (Builder)
class TupleNode: public ExprNode {
public:
    std::vector<std::shared_ptr<ASTNode>> val;

    TupleNode(int line) : ExprNode(line) {};

    std::string toString() override;
    std::vector<std::shared_ptr<ASTNode>> getVal();
};
