#pragma once

#include "ASTNode/ASTNode.h"
#include "ASTNode/Expr/ExprNode.h"

// Children: [ expr ]
class StreamOut : public ASTNode{
public:
    StreamOut(int line);

    std::shared_ptr<ExprNode> getExpr();

    std::string toString() override;
};

