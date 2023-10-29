#pragma once

#include "ASTNode.h"
#include "ASTNode/Expr/ExprNode.h"

class PrintNode : public ASTNode{
public:
    PrintNode(int line);

    std::shared_ptr<ExprNode> getExpr();

    std::string toString() override;
};

