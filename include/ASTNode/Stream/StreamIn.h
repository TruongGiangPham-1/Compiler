#pragma once

#include "ASTNode/ASTNode.h"
#include "ASTNode/Expr/ExprNode.h"

// Children: [ expr ]
class StreamIn : public ASTNode{
public:
    StreamIn(int line);

    std::shared_ptr<ExprNode> getExpr();

    std::string toString() override;
};

